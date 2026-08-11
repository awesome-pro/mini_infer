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
"""

from __future__ import annotations

from collections.abc import Sequence

from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, SchedulerOutput, ScheduledWork
from mini_infer.memory.block_manager import BlockPlan, PagedBlockManager

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
        self.plan = BlockPlan(memory) if memory is not None else None
        self._requirements: dict[str, int] = {}
        self._preempted_this_step: set[str] = set()

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
        self._reset_step_state()

        work: list[ScheduledWork] = []
        self._build_plan(work, waiting, running, budget)
        preemptions = self._evict_to_fund(work, running, waiting)
        if preemptions:
            # Victims have been evicted and their blocks returned, so re-plan to
            # use the freed memory and to guarantee no evicted request still holds
            # work in the plan.
            self._reset_step_state(keep_preempted=True)
            work = []
            self._build_plan(work, waiting, running, budget)

        return SchedulerOutput(work=tuple(work), preemptions=preemptions)

    def _reset_step_state(self, *, keep_preempted: bool = False) -> None:
        self.plan = BlockPlan(self.memory) if self.memory is not None else None
        self._requirements = {}
        if not keep_preempted:
            self._preempted_this_step = set()

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
            if request.is_finished or request.id in self._preempted_this_step:
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
            if request.remaining_prefill == 0 or request.id in self._preempted_this_step:
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
                target = request.sequence_len + chunk
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

        pool = list(running) + list(waiting)
        preemptions: dict[str, str] = {}
        for requester_id, blocks_needed in self._requirements.items():
            if blocks_needed <= memory.num_free_blocks():
                continue
            requester = next((r for r in pool if r.id == requester_id), None)
            if requester is None:  # pragma: no cover - defensive
                continue
            for victim in sorted(pool, key=self.preemption_key):
                if blocks_needed <= memory.num_free_blocks():
                    break
                if not self._can_preempt(victim, requester):
                    continue
                memory.free(victim)
                preemptions[victim.id] = requester_id
                self._preempted_this_step.add(victim.id)
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
        if victim.id in self._preempted_this_step:
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
        needed = self.plan.manager.blocks_needed_to_reach(request, request.sequence_len + 1)
        return max(0, needed - request.block_table.num_blocks)

    def _blocks_for_prefill(self, request: Request) -> int:
        """Blocks the request needs to prefill its whole remaining prompt."""
        assert self.plan is not None
        target = request.sequence_len + request.remaining_prefill
        needed = self.plan.manager.blocks_needed_to_reach(request, target)
        return max(0, needed - request.block_table.num_blocks)

    def _require(self, request_id: str, blocks: int) -> None:
        self._requirements[request_id] = max(
            self._requirements.get(request_id, 0), max(0, blocks)
        )

    def _chunk_size(self, request: Request, remaining_budget: int) -> int:
        """Prefill tokens to grant, clamped by the chunk cap, budget and KV pool."""
        cap = min(request.remaining_prefill, self._chunk_cap(), remaining_budget)
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


class FCFSPolicy(SchedulerBase):
    """First come, first served, with decode before prefill inside a step."""

    name = "fcfs"


def build_policy(
    name: str, config: EngineConfig, *, memory: PagedBlockManager | None = None
) -> SchedulerBase:
    """Instantiate a scheduling policy by name."""
    policies = {FCFSPolicy.name: FCFSPolicy}
    try:
        policy_cls = policies[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; available: {sorted(policies)}") from None
    return policy_cls(config, memory=memory)
