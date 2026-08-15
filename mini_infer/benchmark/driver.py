"""Driving an engine with a workload.

The engine steps; the driver decides when the next request becomes visible. Keeping
that split means arrival policy (burst, open-loop, closed-loop) never leaks into the
engine, and a closed-loop workload can wait for a completion without the engine
knowing anything about concurrency.

The engine's clock is authoritative. Under a ``VirtualClock`` the whole run is
simulated and reproducible; under a ``WallClock`` the same code drives real time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mini_infer.benchmark.workloads import Workload
from mini_infer.engine.engine import Engine, EngineStep
from mini_infer.metrics import RunMetrics


@dataclass(slots=True)
class RunResult:
    """What one engine run produced."""

    workload: Workload
    metrics: RunMetrics
    steps: int
    stalled: bool
    wall_time: float

    @property
    def finished_all(self) -> bool:
        return self.metrics.num_finished == self.metrics.num_requests

    def as_row(self, **labels: object) -> dict[str, object]:
        """Flat record for a results table or CSV."""
        row: dict[str, object] = dict(labels)
        row.update(self.workload.describe())
        row.update(self.metrics.as_row())
        row["stalled"] = self.stalled
        return row


class Driver:
    """Feeds a workload to an engine and runs it to completion."""

    def __init__(
        self,
        engine: Engine,
        workload: Workload,
        *,
        max_steps: int | None = None,
        record_steps: bool = True,
        stall_patience: int = 64,
    ) -> None:
        self.engine = engine
        self.workload = workload
        self.max_steps = max_steps
        self.record_steps = record_steps
        #: Consecutive no-progress steps tolerated before declaring a deadlock. It is
        #: deliberately larger than the engine's own tolerance: under KV pressure a
        #: request can legitimately wait many steps for the eviction that freed its
        #: memory to be committed, and a benchmark must not mistake that for a hang.
        self.stall_patience = stall_patience
        self.steps: list[EngineStep] = []

        self._next_index = 0
        self._submitted = 0
        self._last_arrival = 0.0
        self._scheduled_at: float | None = None

    # ------------------------------------------------------------------ state

    @property
    def pending(self) -> int:
        """Requests the process has not offered yet."""
        return self.workload.num_requests - self._submitted

    @property
    def in_flight(self) -> int:
        """Submitted but not yet finished, including anything still queued."""
        return self._submitted - len(self.engine.finished)

    def peek(self) -> float | None:
        """When the next request becomes visible, or None if no more are coming."""
        if self._scheduled_at is not None:
            return self._scheduled_at
        if self._next_index >= self.workload.num_requests:
            return None
        self._scheduled_at = self.workload.arrivals.next_arrival_time(
            previous_arrival=self._last_arrival,
            completed=len(self.engine.finished),
            in_flight=self.in_flight,
            pending=self.pending,
            num_requests=self.workload.num_requests,
        )
        return self._scheduled_at

    def _submit_next(self, arrival: float) -> None:
        request = self.workload.requests[self._next_index]
        request.arrival_time = arrival
        self.engine.submit(request)
        self._next_index += 1
        self._submitted += 1
        self._last_arrival = arrival
        self._scheduled_at = None

    def coordinate(self, now: float) -> None:
        """Release every request that should be visible at ``now``."""
        while True:
            arrival = self.peek()
            if arrival is None or arrival > now:
                return
            self._submit_next(arrival)

    # -------------------------------------------------------------------- run

    def run(self) -> RunResult:
        started = time.perf_counter()
        consecutive_stalls = 0

        while True:
            if self.max_steps is not None and len(self.steps) >= self.max_steps:
                break

            now = self.engine.clock.now()
            self.coordinate(now)

            arrival = self.peek()
            if arrival is not None and arrival > now and not self.engine.has_pending_work():
                # Nothing can happen before the next request appears, so skip the
                # idle interval rather than stepping through it.
                self.engine.clock.advance(arrival - now)
                self.coordinate(self.engine.clock.now())

            if not self.engine.has_pending_work():
                if self.peek() is None:
                    break
                # A client is waiting on a completion that has already been
                # recorded, so its next request is due now: loop and release it.
                continue

            step = self.engine.step()
            if self.record_steps:
                self.steps.append(step)

            # A stalled step means no request advanced. Distinguish the two cases:
            # requests are still resident, so their completions can free memory and
            # the run should continue; or nothing is running, so no progress is
            # possible and the stall is terminal.
            if step.is_stalled:
                if self.engine.num_running == 0:
                    break
                consecutive_stalls += 1
                if consecutive_stalls > self.stall_patience:
                    break
            else:
                consecutive_stalls = 0

        return self._result(stalled=self._stalled(), started=started)

    def _stalled(self) -> bool:
        """Whether the run ended stuck rather than finished.

        A run that completed every request is never stalled, even though the final
        step schedules nothing: the work is done, not blocked.
        """
        if not self.steps or self.steps[-1].is_stalled is False:
            return False
        return len(self.engine.finished) < self.workload.num_requests

    def _result(self, *, stalled: bool, started: float) -> RunResult:
        elapsed = self.engine.clock.now()
        return RunResult(
            workload=self.workload,
            metrics=self.engine.metrics.summary(elapsed=elapsed),
            steps=len(self.steps),
            stalled=stalled,
            wall_time=time.perf_counter() - started,
        )
