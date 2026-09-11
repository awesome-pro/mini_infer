"""The engine: one step per model forward pass, forever.

The engine owns request state and the clock. The scheduler decides what should
run; the runner performs it; the engine applies the results and decides who is
finished.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from mini_infer.clock import Clock, VirtualClock
from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, SchedulerOutput
from mini_infer.memory.block_manager import PagedBlockManager
from mini_infer.metrics.collector import MetricsCollector
from mini_infer.runner.base import ModelRunner
from mini_infer.runner.simulated_runner import SimulatedModelRunner


class StepEventKind(StrEnum):
    ADMITTED = "admitted"
    PREFILLED = "prefilled"
    DECODED = "decoded"
    PREEMPTED = "preempted"
    PREFIX_CACHED = "prefix_cached"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class StepEvent:
    kind: StepEventKind
    request_id: str
    num_tokens: int = 0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class EngineStep:
    """Record of one engine iteration."""

    index: int
    start_time: float
    end_time: float
    output: SchedulerOutput
    events: tuple[StepEvent, ...] = ()
    free_blocks: int | None = None
    total_blocks: int | None = None
    preemptions: dict[str, str] = field(default_factory=dict)
    running_requests: int = 0
    waiting_requests: int = 0
    progress: tuple = ()
    previous_progress: tuple = ()

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_prefill_tokens(self) -> int:
        return self.output.num_prefill_tokens

    @property
    def num_decode_requests(self) -> int:
        return self.output.num_decode_requests

    @property
    def num_preemptions(self) -> int:
        return len(self.preemptions)

    @property
    def kv_utilization(self) -> float:
        """Share of the KV pool allocated at the end of this step."""
        if self.free_blocks is None or not self.total_blocks:
            return 0.0
        return (self.total_blocks - self.free_blocks) / self.total_blocks

    @property
    def is_stalled(self) -> bool:
        """True when no request made progress this step.

        Distinct from an empty plan: a request whose output budget is spent
        produces no scheduled tokens yet must still be finalised, and a
        zero-progress step that keeps the engine's state identical is what a
        deadlock (exhausted KV before preemption exists) looks like.
        """
        return self.progress == self.previous_progress

    def events_of(self, kind: StepEventKind) -> tuple[StepEvent, ...]:
        return tuple(e for e in self.events if e.kind is kind)


class Scheduler(Protocol):
    """What the engine needs from a scheduler."""

    def schedule(
        self,
        waiting: Sequence[Request],
        running: Sequence[Request],
        *,
        now: float,
        budget: int,
        memory: PagedBlockManager | None = None,
    ) -> SchedulerOutput: ...


class Engine:
    """Coordinates scheduling, KV memory and the model runner."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        clock: Clock | None = None,
        scheduler: Scheduler | None = None,
        runner: ModelRunner | None = None,
        memory: PagedBlockManager | None = None,
        metrics: MetricsCollector | None = None,
        max_stalled_steps: int = 2,
    ) -> None:
        self.config = config or EngineConfig()
        self.clock: Clock = clock if clock is not None else VirtualClock()
        self.runner: ModelRunner = (
            runner if runner is not None else SimulatedModelRunner(self.config.runner)
        )
        self.memory: PagedBlockManager | None = (
            memory
            if memory is not None
            else PagedBlockManager(
                self.config.num_blocks,
                self.config.block_size,
                enable_prefix_cache=self.config.enable_prefix_cache,
            )
        )
        self.metrics: MetricsCollector = metrics if metrics is not None else MetricsCollector()
        self.metrics.attach(self)
        self.max_stalled_steps = max_stalled_steps
        #: True when the runner attends over the physical KV blocks. That decides both
        #: when the engine commits allocations and how it books sampled tokens.
        self.attends_over_kv: bool = bool(getattr(self.runner, "attends_over_kv", False))
        self.scheduler: Scheduler = (
            scheduler
            if scheduler is not None
            else _build_default_scheduler(
                self.config, memory=self.memory, pending_decode_token=self.attends_over_kv
            )
        )
        # The runner is the authority on the token convention: it is the thing that
        # either has logits to sample from or does not. Keep the scheduler in step.
        if hasattr(self.scheduler, "pending_decode_token"):
            self.scheduler.pending_decode_token = self.attends_over_kv

        self.requests: dict[str, Request] = {}
        self.waiting: list[Request] = []
        self.running: dict[str, Request] = {}
        self.finished: list[Request] = []
        #: Every :class:`EngineStep` executed, for timelines and metrics.
        self.history: list[EngineStep] = []
        #: Ids of requests whose completion has been recorded, so a request that
        #: finishes is finalised exactly once even if it is rescheduled later.
        self._recorded_finished: set[str] = set()

        self._arrival_heap: list[tuple[float, int, Request]] = []
        self._arrival_seq = 0
        self._step_index = 0

    # ------------------------------------------------------------------ input

    def submit(self, request: Request) -> Request:
        """Admit a request to the waiting queue. Arrival time is set to now if unset."""
        if request.arrival_time is None or request.arrival_time < self.clock.now():
            request.arrival_time = self.clock.now()
        if request.id in self.requests:
            raise ValueError(f"duplicate request id {request.id!r}")
        self.requests[request.id] = request
        self.waiting.append(request)
        self.metrics.register(request)
        return request

    def submit_at(self, request: Request) -> Request:
        """Schedule a request to become visible to the engine at its arrival time."""
        if request.arrival_time < self.clock.now():
            raise ValueError(
                f"{request.id}: arrival {request.arrival_time} is in the past "
                f"(now={self.clock.now()})"
            )
        if request.id in self.requests:
            raise ValueError(f"duplicate request id {request.id!r}")
        self.metrics.register(request)
        heapq.heappush(self._arrival_heap, (request.arrival_time, self._arrival_seq, request))
        self._arrival_seq += 1
        return request

    def _release_arrivals(self, now: float) -> None:
        while self._arrival_heap and self._arrival_heap[0][0] <= now:
            _, _, request = heapq.heappop(self._arrival_heap)
            self.requests[request.id] = request
            self.waiting.append(request)
            self.metrics.register(request)

    # ------------------------------------------------------------------- step

    @property
    def peak_kv_utilization(self) -> float:
        """Highest KV occupancy seen during the run."""
        if self.memory is None:
            return 0.0
        return self.memory.peak_utilization

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_pending_work(self) -> bool:
        return bool(self.waiting or self.running or self._arrival_heap)

    def _sync_running_set(self) -> None:
        """Refresh the running set from request status.

        Status is the single source of truth (a policy may preempt a request
        mid-schedule), so the engine derives its running set instead of owning it.
        """
        for request in tuple(self.running.values()):
            if not request.status.is_active:
                del self.running[request.id]
        for request in self.waiting:
            if request.status.is_active:
                self.running[request.id] = request
        if self.running:
            active_ids = set(self.running)
            self.waiting = [r for r in self.waiting if r.id not in active_ids]

    def step(self) -> EngineStep:
        """Run exactly one engine iteration (one model forward pass)."""
        self._release_arrivals(self.clock.now())
        self._sync_running_set()
        now = self.clock.now()
        start = now
        progress_before = self._progress_fingerprint()

        output = self.scheduler.schedule(
            self.waiting,
            list(self.running.values()),
            now=now,
            budget=self.config.max_batch_tokens,
            memory=self.memory,
        )

        # Attachments come first: the runner reads the shared blocks, and the cursor
        # has to cover them before anything measures a context length.
        prefix_events = self._commit_prefix_attachments(output)

        context_lengths = {
            work.request_id: work.request.num_computed_tokens + work.num_new_tokens
            for work in output.work
        }

        # A runner that attends over the KV blocks cannot compute without them, so its
        # allocations are committed first. The plan was funded by the victims it chose,
        # so those are released before the allocation. The commit order below is the
        # speculative one, for runners that only report a duration.
        preemptions: list[StepEvent] = []
        if self.attends_over_kv:
            preemptions = self._release_preempted_kv(output.preemptions)
            self._allocate_kv_for_work(output)

        timing = self.runner.time_step(output, context_lengths=context_lengths)
        sampled = self.runner.execute(output, context_lengths=context_lengths)

        self.clock.advance(timing.duration_s)
        end = self.clock.now()

        events = list(prefix_events)
        events.extend(self._apply(output, sampled.sampled_tokens, end))
        # The cursor now covers positions whose KV the runner produced, so their blocks
        # are safe to share. A simulated runner reaches this through _allocate_kv; a real
        # one allocates before running, so it needs this explicit point.
        if self.memory is not None:
            for work in output.work:
                self.memory.cache_full_blocks(work.request)
        # Commit order for a simulated runner: free the victims' KV and rewind them,
        # release anyone who finished, then allocate for the work that ran. The
        # scheduler planned all of this without touching memory, so this is the only
        # place state moves.
        events.extend(
            preemptions
            if self.attends_over_kv
            else self._release_preempted_kv(output.preemptions)
        )
        self._release_finished_kv()
        if not self.attends_over_kv:
            self._allocate_kv(output)

        step = EngineStep(
            index=self._step_index,
            start_time=start,
            end_time=end,
            output=output,
            events=tuple(events),
            free_blocks=self.memory.num_free_blocks() if self.memory else None,
            total_blocks=self.memory.num_blocks if self.memory else None,
            preemptions=dict(output.preemptions),
            running_requests=len(self.running),
            waiting_requests=len(self.waiting),
            progress=self._progress_fingerprint(),
            previous_progress=progress_before,
        )
        self._step_index += 1
        self.history.append(step)
        self.metrics.record_step(step)
        # Leave the running set consistent with request status, so callers
        # inspecting the engine between steps see the truth.
        self._sync_running_set()
        return step

    def run(
        self, arrivals: Iterable[Request] | None = None, *, max_steps: int | None = None
    ) -> Iterator[EngineStep]:
        """Run until every request finishes, yielding each step.

        Requests supplied in ``arrivals`` are released by arrival time; requests
        already submitted are picked up immediately. When the engine is idle it
        jumps the clock to the next arrival instead of spinning. A scheduler that
        keeps returning empty steps while work is queued would livelock, so the
        loop stops and leaves the queue intact for inspection.
        """
        for request in arrivals or ():
            self.submit_at(request)

        stalled_steps = 0
        while self.has_pending_work():
            if max_steps is not None and self._step_index >= max_steps:
                return
            if not self.waiting and not self.running and self._arrival_heap:
                self.clock.advance(max(0.0, self._arrival_heap[0][0] - self.clock.now()))
            step = self.step()
            if step.is_stalled:
                # Nothing ran and no state changed. A request blocked on KV may be
                # freed by another request finishing, so tolerate a couple of
                # steps; beyond that the engine is deadlocked (before preemption
                # exists) and the run stops with the stall visible in the last step
                # rather than looking like a completed workload.
                stalled_steps += 1
                if stalled_steps > self.max_stalled_steps:
                    return
            else:
                stalled_steps = 0
            yield step

    def run_to_completion(
        self, arrivals: Iterable[Request] | None = None, *, max_steps: int | None = None
    ) -> list[EngineStep]:
        return list(self.run(arrivals, max_steps=max_steps))

    # ---------------------------------------------------------------- results

    def _commit_prefix_attachments(self, output: SchedulerOutput) -> list[StepEvent]:
        """Share the cached prefix blocks the plan decided each request should inherit.

        The scheduler is the only thing that decides this, exactly as it decides
        allocations: a hit shrinks the request's remaining work and its block
        requirement, and admission control sees the shared blocks in the pool. The
        engine's part is to refcount them and move the cursor past them, before the
        runner — which needs them — and before any allocation, so a block the plan
        promised to a request cannot be evicted out from under it in the same step.

        Only the blocks themselves are shared, not a claim on them: a full cached block
        is immutable, so two requests reading it cannot disturb each other.
        """
        memory = self.memory
        if memory is None or not output.attachments:
            return []

        events: list[StepEvent] = []
        for request_id, blocks in output.attachments.items():
            request = self.requests.get(request_id)
            if request is None:  # pragma: no cover - defensive
                continue
            position, _ = memory.cached_prefix(request)
            held = request.num_computed_tokens
            if position <= held:  # pragma: no cover - the pool changed under the plan
                continue
            memory.attach_prefix(request, blocks)
            request.on_prefill(position - held)
            events.append(
                StepEvent(
                    StepEventKind.PREFIX_CACHED,
                    request_id,
                    position - held,
                    detail=f"shared {len(blocks)} blocks, skipped {position - held} tokens",
                )
            )
        return events

    def _apply(
        self,
        output: SchedulerOutput,
        sampled_tokens: dict[str, int],
        now: float,
    ) -> list[StepEvent]:
        if self.attends_over_kv:
            return self._apply_sampled(output, sampled_tokens, now)

        events: list[StepEvent] = []
        finished: list[Request] = []

        for work in output.work:
            request = work.request
            prior_status = request.status

            if prior_status is RequestStatus.WAITING:
                request.on_admitted(self._step_index)
                events.append(StepEvent(StepEventKind.ADMITTED, request.id))

            if work.kind is ScheduledKind.PREFILL:
                request.on_prefill(work.num_new_tokens)
                events.append(
                    StepEvent(StepEventKind.PREFILLED, request.id, work.num_new_tokens)
                )
                if request.is_complete:
                    # Prompt cached and no output budget left: finish now rather
                    # than entering a decode state that can never complete.
                    request.mark_complete()

                if request.is_finished and request.id not in self._recorded_finished:
                    request.on_finished(now)
                    finished.append(request)
                    events.append(
                        StepEvent(StepEventKind.FINISHED, request.id, detail="no output budget")
                    )
                continue

            if work.num_new_tokens == 0:
                # Zero tokens claims a slot without executing: the KV pool could
                # not fund this work in this step.
                continue

            request.on_decode(sampled_tokens.get(request.id, 0), now)
            events.append(StepEvent(StepEventKind.DECODED, request.id, 1))
            if request.is_complete:
                request.mark_complete()

            if request.is_finished and request.id not in self._recorded_finished:
                request.on_finished(now)
                finished.append(request)
                events.append(
                    StepEvent(
                        StepEventKind.FINISHED, request.id, detail="max_new_tokens reached"
                    )
                )

        for request in finished:
            self.running.pop(request.id, None)
            self._recorded_finished.add(request.id)
            self.finished.append(request)

        if finished:
            finished_ids = {r.id for r in finished}
            self.waiting = [r for r in self.waiting if r.id not in finished_ids]

        return events

    def _apply_sampled(
        self,
        output: SchedulerOutput,
        sampled_tokens: dict[str, int],
        now: float,
    ) -> list[StepEvent]:
        """Commit a step from a runner whose logits produced the tokens.

        The cursor advances over the positions the runner was granted, exactly as in
        :meth:`_apply`. The difference is where the token comes from: a real forward
        samples the *next* token while computing those positions, so the token is
        appended behind the cursor and its own position stays pending for the next
        step. The first token therefore lands at the end of prefill — which is when a
        server really has something to stream, and so where time-to-first-token should
        be measured.
        """
        events: list[StepEvent] = []
        finished: list[Request] = []

        for work in output.work:
            request = work.request

            if request.status is RequestStatus.WAITING:
                request.on_admitted(self._step_index)
                events.append(StepEvent(StepEventKind.ADMITTED, request.id))

            if work.num_new_tokens <= 0:
                # Zero tokens claims a slot without executing: the KV pool could not
                # fund this work in this step.
                continue

            request.on_prefill(work.num_new_tokens)
            if work.kind is ScheduledKind.PREFILL:
                events.append(
                    StepEvent(StepEventKind.PREFILLED, request.id, work.num_new_tokens)
                )

            # The forward covered the whole sequence, so its last row is the next
            # token for this request.
            if request.num_computed_tokens == request.num_tokens and not request.is_complete:
                token = sampled_tokens.get(request.id)
                if token is not None:
                    request.on_sampled(token, now)
                    events.append(StepEvent(StepEventKind.DECODED, request.id, 1))

            if request.is_complete:
                request.mark_complete()

            if request.is_finished and request.id not in self._recorded_finished:
                request.on_finished(now)
                finished.append(request)
                events.append(
                    StepEvent(
                        StepEventKind.FINISHED, request.id, detail="max_new_tokens reached"
                    )
                )

        for request in finished:
            self.running.pop(request.id, None)
            self._recorded_finished.add(request.id)
            self.finished.append(request)

        if finished:
            finished_ids = {r.id for r in finished}
            self.waiting = [r for r in self.waiting if r.id not in finished_ids]

        return events

    # ----------------------------------------------------------------- memory

    def _release_preempted_kv(self, preemptions: dict[str, str]) -> list[StepEvent]:
        """Commit evictions: free the victims' KV and rewind them for recomputation.

        The scheduler only *decided* to evict; this is where the physical blocks
        actually return to the pool and the victim's cursor resets, so a failed or
        abandoned plan never leaves memory moved.
        """
        if self.memory is None:
            return []

        events: list[StepEvent] = []
        for victim_id, requester_id in preemptions.items():
            victim = self.requests.get(victim_id)
            if victim is None:  # pragma: no cover - defensive
                continue
            self.running.pop(victim_id, None)
            self.memory.free(victim)
            victim.on_preempted()
            if all(r.id != victim_id for r in self.waiting):
                self.waiting.append(victim)
            events.append(
                StepEvent(
                    StepEventKind.PREEMPTED,
                    victim_id,
                    detail=f"evicted for {requester_id}",
                )
            )
        return events

    def _release_finished_kv(self) -> None:
        """Return the KV of every finished request to the pool, exactly once.

        Done as its own phase so the memory released here is available to the
        allocations that immediately follow in the same step.
        """
        if self.memory is None:
            return
        for request in self.finished:
            if request.block_table.num_blocks > 0:
                self.memory.free(request)

    def _progress_fingerprint(self) -> tuple:
        """Cheap snapshot of everything a step is supposed to advance.

        A request admitted with zero prefill tokens still changes its status, so it
        counts as progress; a step that leaves every request untouched does not.
        """
        return (
            len(self._arrival_heap),
            len(self.waiting),
            len(self.running),
            len(self.finished),
            tuple(
                sorted(
                    (r.id, r.status.value, r.num_computed_tokens, r.num_generated)
                    for r in self.requests.values()
                )
            ),
        )

    def _allocate_kv(self, output: SchedulerOutput) -> None:
        """Commit this step's block allocations.

        The plan already decided how many extra blocks each request needs, so the
        engine only has to hand them out. Two kinds of request are skipped: one that
        finished inside this step and one that was evicted during it. In both cases
        its KV is released in this same step, so allocating would only force the
        pool to over-commit memory that is already dead.
        """
        if self.memory is None:
            return
        for request_id, extra_blocks in output.allocations.items():
            if extra_blocks <= 0:
                continue
            request = self.requests.get(request_id)
            # Only a request that is still active holds KV after this step.
            if request is None or not request.status.is_active:
                continue
            # `extra_blocks` is exactly the shortfall between the blocks held and
            # the blocks needed to cover the cursor, so growing to the cursor takes
            # precisely those blocks.
            self.memory.grow_to(request, request.num_computed_tokens)

    def _allocate_kv_for_work(self, output: SchedulerOutput) -> None:
        """Commit the blocks for work that has not run yet.

        Used only when the runner attends over the KV blocks: it writes into the
        positions it was granted, so those blocks must exist before it executes. The
        target is the cursor *after* the grant, which is what the runner will fill in;
        by the time the step is applied the cursor has caught up, so the pool's
        invariants hold again.
        """
        if self.memory is None:
            return
        for work in output.work:
            if work.num_new_tokens <= 0:
                continue
            request = work.request
            # Anything still holding work here is about to compute, including a
            # preempted request being recomputed: it is not active yet, but it is
            # about to be, and it needs blocks to hold the KV it will write.
            if request.status is RequestStatus.FINISHED:  # pragma: no cover - defensive
                continue
            self.memory.grow_to(request, request.num_computed_tokens + work.num_new_tokens)


def _build_default_scheduler(
    config: EngineConfig,
    *,
    memory: PagedBlockManager | None,
    pending_decode_token: bool = False,
) -> Scheduler:
    from mini_infer.engine.policies import build_policy

    return build_policy(
        config.policy, config, memory=memory, pending_decode_token=pending_decode_token
    )
