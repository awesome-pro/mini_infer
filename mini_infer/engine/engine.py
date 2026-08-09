"""The engine: one step per model forward pass, forever.

The engine owns request state and the clock. The scheduler decides what should
run; the runner performs it; the engine applies the results and decides who is
finished.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from mini_infer.clock import Clock, VirtualClock
from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, SchedulerOutput
from mini_infer.runner.base import ModelRunner
from mini_infer.runner.simulated_runner import SimulatedModelRunner


class StepEventKind(str, Enum):
    ADMITTED = "admitted"
    PREFILLED = "prefilled"
    DECODED = "decoded"
    PREEMPTED = "preempted"
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

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_prefill_tokens(self) -> int:
        return self.output.num_prefill_tokens

    @property
    def num_decode_requests(self) -> int:
        return self.output.num_decode_requests

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
        idle_step_s: float = 1e-3,
    ) -> None:
        self.config = config or EngineConfig()
        self.clock: Clock = clock if clock is not None else VirtualClock()
        self.runner: ModelRunner = (
            runner if runner is not None else SimulatedModelRunner(self.config.runner)
        )
        self.scheduler: Scheduler = (
            scheduler if scheduler is not None else _build_default_scheduler(self.config)
        )
        self.idle_step_s = idle_step_s

        self.requests: dict[str, Request] = {}
        self.waiting: list[Request] = []
        self.running: dict[str, Request] = {}
        self.finished: list[Request] = []

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
        heapq.heappush(self._arrival_heap, (request.arrival_time, self._arrival_seq, request))
        self._arrival_seq += 1
        return request

    def _release_arrivals(self, now: float) -> None:
        while self._arrival_heap and self._arrival_heap[0][0] <= now:
            _, _, request = heapq.heappop(self._arrival_heap)
            self.requests[request.id] = request
            self.waiting.append(request)

    # ------------------------------------------------------------------- step

    def has_pending_work(self) -> bool:
        return bool(self.waiting or self.running or self._arrival_heap)

    def _sync_running(self) -> None:
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
        self._sync_running()
        now = self.clock.now()
        start = now

        output = self.scheduler.schedule(
            self.waiting,
            list(self.running.values()),
            now=now,
            budget=self.config.max_batch_tokens,
        )

        context_lengths = {
            work.request_id: work.request.sequence_len + work.num_new_tokens
            for work in output.work
        }
        timing = self.runner.time_step(output, context_lengths=context_lengths)
        sampled = self.runner.execute(output, context_lengths=context_lengths)

        self.clock.advance(timing.duration_s)
        end = self.clock.now()
        events = self._apply(output, sampled.sampled_tokens, end)

        step = EngineStep(
            index=self._step_index,
            start_time=start,
            end_time=end,
            output=output,
            events=tuple(events),
        )
        self._step_index += 1
        # Leave the running set consistent with request status, so callers
        # inspecting the engine between steps see the truth.
        self._sync_running()
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

        while self.has_pending_work():
            if max_steps is not None and self._step_index >= max_steps:
                return
            if not self.waiting and not self.running and self._arrival_heap:
                self.clock.advance(max(0.0, self._arrival_heap[0][0] - self.clock.now()))
            step = self.step()
            if step.output.is_empty and not step.events:
                if not self._arrival_heap or not self.waiting:
                    return
            yield step

    def run_to_completion(
        self, arrivals: Iterable[Request] | None = None, *, max_steps: int | None = None
    ) -> list[EngineStep]:
        return list(self.run(arrivals, max_steps=max_steps))

    # ---------------------------------------------------------------- results

    def _apply(
        self,
        output: SchedulerOutput,
        sampled_tokens: dict[str, int],
        now: float,
    ) -> list[StepEvent]:
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
                continue

            request.on_decode(sampled_tokens.get(request.id, 0), now)
            events.append(StepEvent(StepEventKind.DECODED, request.id, 1))
            if request.is_complete:
                request.on_finished(now)
                finished.append(request)
                events.append(
                    StepEvent(
                        StepEventKind.FINISHED, request.id, detail="max_new_tokens reached"
                    )
                )

        for request in finished:
            self.running.pop(request.id, None)
            self.finished.append(request)

        if finished:
            finished_ids = {r.id for r in finished}
            self.waiting = [r for r in self.waiting if r.id not in finished_ids]

        return events


def _build_default_scheduler(config: EngineConfig) -> Scheduler:
    from mini_infer.engine.policies import build_policy

    return build_policy(config.policy, config)
