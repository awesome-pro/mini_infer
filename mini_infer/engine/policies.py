"""Scheduling policies.

Two ideas shape every policy here:

* **The token budget is a hard ceiling.** Every step grants at most
  ``max_batch_tokens`` of work, split between decode (one token per request) and
  prefill chunks.
* **Memory is consulted while planning, not afterwards.** The plan is built
  against the blocks that are free, and if it does not fit, a victim is evicted
  and the plan is rebuilt. That ordering keeps the plan and the pool consistent:
  a request is never both evicted and executed in the same step.

Chunked prefill is a knob rather than a policy: with it off, a prompt that fits a
step is prefilled in one shot, and one that does not is still split, because the
budget is a hard limit and refusing to split would deadlock.

Every policy answers the same question differently: *who gets this step's token
budget?* That is deliberate, because it isolates one decision at a time.

``fcfs`` / ``decode_first``
    Decodes are served first, prefills take what is left. The two are identical by
    construction — serving the decoding set before the prefill set *is* decode
    priority — and ``decode_first`` exists as a name so a comparison can point at
    the pole it is measuring.
``prefill_first``
    The opposite order: prefills first, so time-to-first-token is low and the
    decoding requests pay for it.
``balanced``
    Decode-first with a prefill floor (``prefill_reservation``) and ageing
    (``max_wait_steps``), so that neither side can be starved.
``static``
    Not a budget decision but an admission one: a batch forms and no new request
    joins it until the batch has drained. The baseline that continuous batching
    replaces.
"""

from __future__ import annotations

from collections.abc import Sequence

from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, ScheduledWork, SchedulerOutput
from mini_infer.memory.block_manager import (
    BlockPlan,
    PagedBlockManager,
    blocks_for_tokens,
)

PHASE_ORDER = ("decode", "prefill")


class SchedulerBase:
    """Budget and memory bookkeeping shared by every policy.

    Subclasses decide the order of the two phases and may reserve part of the
    step budget for prefills.
    """

    name = "base"
    phase_order: tuple[str, ...] = PHASE_ORDER

    def __init__(self, config: EngineConfig, memory: PagedBlockManager | None = None) -> None:
        self.config = config
        self.memory = memory
        #: Transactional view of the KV pool for the step being planned.
        self.plan: BlockPlan | None = BlockPlan(memory) if memory is not None else None
        self._requirements: dict[str, int] = {}
        #: Consecutive steps each request has spent waiting for its first chunk.
        self._wait_steps: dict[str, int] = {}

    # ------------------------------------------------------------- interfaces

    def schedule(
        self,
        waiting: Sequence[Request],
        running: Sequence[Request],
        *,
        now: float,
        budget: int,
        memory: PagedBlockManager | None = None,
    ) -> SchedulerOutput:
        self.memory = memory if memory is not None else self.memory
        self._track_waiting(waiting)
        self._reset_step_state()

        work: list[ScheduledWork] = []
        self._build_plan(work, waiting, running, budget)
        preemptions = self._evict_to_fund(work, running, waiting)
        if preemptions:
            # The plan now knows which blocks those evictions free, so re-planning
            # uses them. Nothing has been committed yet: the engine applies the
            # frees and the allocation together, after the runner has succeeded.
            self._reset_step_state()
            self.plan = BlockPlan(self.memory, victims=self.plan.victims())
            work = []
            self._build_plan(work, waiting, running, budget)

        return SchedulerOutput(
            work=tuple(work),
            preemptions=preemptions,
            allocations=self.plan.claims() if self.plan is not None else {},
        )

    def _reset_step_state(self) -> None:
        self.plan = BlockPlan(self.memory) if self.memory is not None else None
        self._requirements = {}

    def _track_waiting(self, waiting: Sequence[Request]) -> None:
        """Count how many consecutive steps each request has spent waiting.

        One call per engine step, so the count is in steps rather than milliseconds
        and stays meaningful under a virtual clock. A request that leaves the queue
        is forgotten, so a later re-admission starts from zero.
        """
        present = set()
        for request in waiting:
            present.add(request.id)
            self._wait_steps[request.id] = self._wait_steps.get(request.id, 0) + 1
        for request_id in [i for i in self._wait_steps if i not in present]:
            del self._wait_steps[request_id]

    def waited_steps(self, request: Request) -> int:
        """Consecutive steps this request has waited for its first chunk."""
        return self._wait_steps.get(request.id, 0)

    def _build_plan(
        self,
        work: list[ScheduledWork],
        waiting: Sequence[Request],
        running: Sequence[Request],
        budget: int,
    ) -> None:
        remaining = budget
        for phase in self.phase_order:
            if phase == "decode":
                remaining -= self._schedule_decode(work, running, remaining)
            else:
                remaining -= self._schedule_prefill(work, waiting, running, remaining)

    # ---------------------------------------------------------------- phases

    def _schedule_decode(
        self,
        work: list[ScheduledWork],
        running: Sequence[Request],
        budget: int,
    ) -> int:
        """Plan one token per decoding request, as far as memory allows.

        A request that cannot be funded keeps its slot with a zero-token grant, and
        its block requirement is recorded for the eviction pass.
        """
        plan = self.plan
        used = 0

        for request in sorted(running, key=self.decode_order_key):
            if len(work) >= self.config.max_running_requests:
                break
            if request.status is not RequestStatus.DECODING:
                # A request still mid-prompt is served by the prefill phase.
                continue
            if request.is_finished or (plan is not None and plan.is_evicted(request.id)):
                continue
            if request.is_complete:
                # Output budget already spent: hand it to the engine to finalise
                # without spending compute on it.
                work.append(
                    ScheduledWork(request=request, num_new_tokens=0, kind=ScheduledKind.DECODE)
                )
                continue
            if used + 1 > budget:
                break

            if plan is not None:
                extra = self._decode_extra_blocks(request)
                if not plan.can_fit(request, extra):
                    # Not fundable from the memory that is free. Record what it
                    # needs so the eviction pass can free a victim, and leave it out
                    # of the plan: reserving a slot for work that cannot run would
                    # make the plan disagree with the pool.
                    self._require(request.id, extra)
                    continue
                plan.claim(request.id, extra)
            work.append(
                ScheduledWork(request=request, num_new_tokens=1, kind=ScheduledKind.DECODE)
            )
            used += 1
        return used

    def _schedule_prefill(
        self,
        work: list[ScheduledWork],
        waiting: Sequence[Request],
        running: Sequence[Request],
        budget: int,
    ) -> int:
        used = 0
        # A request mid-prompt lives in the running set, so both sets are
        # candidates: waiting requests can be admitted, running ones need their
        # next chunk. Once active, a request is no longer in `waiting`.
        active = {r.id for r in running if r.status.is_active}
        candidates = list(waiting) + list(running)
        occupied = len(active)

        for request in sorted(candidates, key=self.prefill_order_key):
            if request.num_uncomputed_tokens == 0 or (
                self.plan is not None and self.plan.is_evicted(request.id)
            ):
                continue
            new_admission = request.id not in active
            if new_admission and occupied >= self.config.max_running_requests:
                continue

            chunk = self._chunk_size(request, budget - used)
            if chunk <= 0:
                # Not fundable right now. A waiting request records its whole prompt
                # as the requirement so the eviction pass can free room for it; an
                # in-flight request waits for memory to free naturally, since
                # evicting for one would let a long prompt thrash against itself.
                if new_admission:
                    self._require(request.id, self._blocks_for_prefill(request))
                continue

            if new_admission:
                occupied += 1
            work.append(
                ScheduledWork(request=request, num_new_tokens=chunk, kind=ScheduledKind.PREFILL)
            )
            used += chunk
            if self.plan is not None:
                target = request.num_computed_tokens + chunk
                needed = self.plan.manager.blocks_needed_to_reach(request, target)
                self.plan.claim(
                    request.id, max(0, needed - request.block_table.num_blocks)
                )
        return used

    # -------------------------------------------------------------- preemption

    def _evict_to_fund(
        self,
        work: list[ScheduledWork],
        running: Sequence[Request],
        waiting: Sequence[Request],
    ) -> dict[str, str]:
        """Evict requests until the pool can fund every block the plan needs.

        Victims are chosen in policy order and their committed work is dropped, so
        an evicted request is never also executed in the same step. Its prefill is
        recomputed when it is next admitted.
        """
        memory = self.memory
        if memory is None or not self.config.enable_preemption:
            return {}

        assert self.plan is not None
        pool = {r.id: r for r in list(running) + list(waiting)}
        preemptions: dict[str, str] = {}
        # Requests still needing funding. A victim stops needing memory and its
        # blocks count towards every later requirement.
        remaining = {
            req_id: blocks
            for req_id, blocks in self._requirements.items()
            if not self.plan.is_evicted(req_id)
        }

        for requester_id, blocks_needed in list(remaining.items()):
            requester = pool.get(requester_id)
            if requester is None:  # pragma: no cover - defensive
                continue
            for victim in sorted(pool.values(), key=self.preemption_key):
                if blocks_needed <= self.plan.free_blocks(requester_id):
                    break
                if not self._can_preempt(victim, requester):
                    continue
                freed = blocks_for_tokens(
                    victim.num_computed_tokens, self.plan.block_size
                )
                if freed <= 0:
                    continue
                self.plan.evict(victim.id, freed)
                preemptions[victim.id] = requester_id
                remaining.pop(victim.id, None)
        if not preemptions:
            return {}
        work[:] = [item for item in work if item.request_id not in preemptions]
        return preemptions

    def _can_preempt(self, victim: Request, requester: Request) -> bool:
        """Whether this victim may be evicted for this requester."""
        if victim.id == requester.id:
            return False
        if victim.status not in (RequestStatus.DECODING, RequestStatus.PREFILLING):
            return False
        assert self.plan is not None
        if self.plan.is_evicted(victim.id):
            return False
        return self._evictable(victim, requester)

    def _evictable(self, victim: Request, requester: Request) -> bool:
        """Evict the request that is youngest in policy order.

        This is the LIFO/RADIO rule: the most recently admitted request gives up
        its KV first, so the requests closest to finishing keep their memory. It is
        a total order, which is what makes the choice converge instead of letting
        two requests hand the pool back and forth.

        A victim that already had work committed this step is fine to evict: the
        plan is rebuilt after eviction and the evicted request is excluded from it,
        so no work is ever left unfunded.
        """
        # Only an older request may take memory from a younger one. Without this a
        # late arrival could evict work that is further along, then be unable to
        # fund its own prefill, and the pool would go back and forth without either
        # request finishing.
        if requester.arrival_time > victim.arrival_time:
            return False
        return self.preemption_key(victim) < self.preemption_key(requester)

    def preemption_key(self, request: Request) -> tuple:
        """Sort key for victims: latest arrival first, so the youngest goes."""
        return (-request.arrival_time, request.id)

    # ------------------------------------------------------------------ hooks

    def _decode_extra_blocks(self, request: Request) -> int:
        """Blocks the request's next decode token needs beyond what it holds."""
        assert self.plan is not None
        needed = self.plan.manager.blocks_needed_to_reach(
            request, request.num_tokens + 1
        )
        return max(0, needed - request.block_table.num_blocks)

    def _blocks_for_prefill(self, request: Request) -> int:
        """Blocks the request needs to prefill its whole remaining prompt."""
        assert self.plan is not None
        target = request.num_tokens
        needed = self.plan.manager.blocks_needed_to_reach(request, target)
        return max(0, needed - request.block_table.num_blocks)

    def _require(self, request_id: str, blocks: int) -> None:
        self._requirements[request_id] = max(
            self._requirements.get(request_id, 0), max(0, blocks)
        )

    def _chunk_size(self, request: Request, remaining_budget: int) -> int:
        """Prefill tokens to grant, clamped by the chunk cap, budget and KV pool."""
        cap = min(request.num_uncomputed_tokens, self._chunk_cap(), remaining_budget)
        if self.plan is None:
            return max(0, cap)
        # The cap passed down must be the policy's own limit, never a chunk already
        # trimmed by memory: trimming first would hide the memory constraint.
        return self.plan.max_growth_tokens(request, cap)

    def _chunk_cap(self) -> int:
        """Largest prefill chunk this policy will grant.

        The step budget is always the hard ceiling. Chunking only lowers it: with
        chunking disabled a prompt that fits the budget is prefilled in one shot,
        while with chunking enabled it is split at ``max_prefill_chunk``.
        """
        if not self.config.enable_chunked_prefill:
            return self.config.max_batch_tokens
        cap = self.config.max_prefill_chunk
        if cap is None:
            cap = self.config.max_batch_tokens
        return max(1, min(cap, self.config.max_batch_tokens))

    def prefill_order_key(self, request: Request) -> tuple:
        return (request.arrival_time, request.id)

    def decode_order_key(self, request: Request) -> tuple:
        return (request.arrival_time, request.id)


class DecodeFirstPolicy(SchedulerBase):
    """Decodes are served first; prefills get whatever budget is left.

    Protects inter-token latency at the cost of time-to-first-token: under a
    decode-heavy load a prefill can be granted zero tokens for the step, and a
    request can wait many steps for its first token. That is the failure
    :class:`BalancedPolicy` bounds, so this policy is worth keeping unpatched.

    It delays rather than deadlocks: admission is ordered by arrival and outputs are
    finite, so the decoding set always drains and frees budget eventually.
    """

    name = "decode_first"


class FCFSPolicy(DecodeFirstPolicy):
    """First come, first served within each phase, decode phase first.

    Both phases order by arrival, so this is the oldest-request-first schedule.
    """

    name = "fcfs"


class PrefillFirstPolicy(SchedulerBase):
    """Prefills are served first; decodes get whatever budget is left.

    Good time-to-first-token and poor inter-token latency: one long prompt can take
    the whole step and push every decode out by a step.
    """

    name = "prefill_first"
    phase_order = ("prefill", "decode")


class BalancedPolicy(SchedulerBase):
    """Decode-first with a prefill floor, plus ageing for the oldest waiter.

    Two mechanisms, each with one job:

    * ``prefill_reservation`` is a floor, in tokens per step, that the decodes may
      not spend. Prefills keep progressing under a decode-heavy load, which is what
      stops them being granted zero tokens step after step.
    * ``max_wait_steps`` decides *which* waiter the floor is spent on: a request that
      has waited this many steps sorts ahead of younger ones in the prefill order,
      and when the reservation is zero it creates a one-token floor anyway (``0``
      disables ageing).

    The wait this bounds is only as tight as the queue allows: a burst of requests
    can all cross the ageing threshold on the same step, and they are then served
    oldest-first, so the worst wait also depends on how many got there first.

    The floor is a floor, not a cap: once the decodes have their share, prefills
    may spend the rest of the step. Both mechanisms act through one schedule, so
    there is no path that starves the decodes to rescue a prefill.
    """

    name = "balanced"

    def _build_plan(
        self,
        work: list[ScheduledWork],
        waiting: Sequence[Request],
        running: Sequence[Request],
        budget: int,
    ) -> None:
        reserve = self._reserve(waiting, running, budget)
        used = self._schedule_decode(work, running, budget - reserve)
        self._schedule_prefill(work, waiting, running, budget - used)

    def _reserve(
        self, waiting: Sequence[Request], running: Sequence[Request], budget: int
    ) -> int:
        """Tokens of this step's budget that only prefills may spend."""
        if not self._prefill_pending(waiting, running):
            return 0
        reserve = min(self.config.prefill_reservation, budget)
        if reserve <= 0 and any(self._is_aged(request) for request in waiting):
            reserve = 1
        if any(request.status is RequestStatus.DECODING for request in running):
            # Never take the last token from the decodes: a reservation equal to the
            # budget would leave them with nothing to do, every step, forever.
            reserve = min(reserve, budget - 1)
        return max(0, reserve)

    def _prefill_pending(self, waiting: Sequence[Request], running: Sequence[Request]) -> bool:
        """Whether a prefill could spend the reservation this step.

        Reserving budget no prefill can use would tax the decodes for nothing, so a
        request that is memory-blocked, or that has no free admission slot, is not a
        reason to reserve.
        """
        active = sum(1 for request in running if request.status.is_active)
        open_slots = max(0, self.config.max_running_requests - active)
        for request in (*running, *waiting):
            if request.num_uncomputed_tokens <= 0:
                continue
            if not request.status.is_active and open_slots <= 0:
                continue
            if self.plan is None or self.plan.max_growth_tokens(request, 1) > 0:
                return True
        return False

    def _is_aged(self, request: Request) -> bool:
        """Whether this request has waited past the ageing bound."""
        limit = self.config.max_wait_steps
        return (
            limit > 0
            and request.num_uncomputed_tokens > 0
            and self.waited_steps(request) >= limit
        )

    def prefill_order_key(self, request: Request) -> tuple:
        # An aged request is served before younger prefills, so the floor goes to
        # the request that has waited longest rather than to a fresh arrival.
        return (0 if self._is_aged(request) else 1, request.arrival_time, request.id)


class StaticBatchPolicy(SchedulerBase):
    """Static batching: a batch forms, then no new request joins until it drains.

    The same engine, budget and memory manager as the continuous policies; only
    admission differs, so comparing it isolates the cost of refusing to refill a
    slot as soon as one frees. Inside a batch prefill and decode still interleave,
    and prefills are still chunked.
    """

    name = "static"

    def _schedule_prefill(
        self,
        work: list[ScheduledWork],
        waiting: Sequence[Request],
        running: Sequence[Request],
        budget: int,
    ) -> int:
        if running:
            # A batch is in flight: arrivals wait for the next batch. Mid-prompt
            # members of the batch still reach the prefill phase through `running`.
            waiting = ()
        return super()._schedule_prefill(work, waiting, running, budget)


#: Every registered policy, keyed by the name a configuration refers to.
POLICIES: dict[str, type[SchedulerBase]] = {
    policy.name: policy
    for policy in (
        FCFSPolicy,
        DecodeFirstPolicy,
        PrefillFirstPolicy,
        BalancedPolicy,
        StaticBatchPolicy,
    )
}


def policy_names() -> tuple[str, ...]:
    """Registered policy names, in a stable order."""
    return tuple(sorted(POLICIES))


def build_policy(
    name: str, config: EngineConfig, *, memory: PagedBlockManager | None = None
) -> SchedulerBase:
    """Instantiate a scheduling policy by name."""
    try:
        policy_cls = POLICIES[name]
    except KeyError:
        raise ValueError(
            f"unknown policy {name!r}; available: {', '.join(policy_names())}"
        ) from None
    return policy_cls(config, memory=memory)
