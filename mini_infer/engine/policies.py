"""Scheduling policies.

Phase 1 ships FCFS with a hard token budget. Chunked prefill is a policy knob:
with it disabled a request is only admitted once its whole remaining prompt fits
the step budget, which is what makes static-style batching reproducible for
comparison. The remaining policies arrive in later phases.
"""

from __future__ import annotations

from collections.abc import Sequence

from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, SchedulerOutput, ScheduledWork

PHASE_ORDER = ("decode", "prefill")


class SchedulerBase:
    """Budget bookkeeping plus the admission scan shared by every policy.

    Subclasses decide the order of the two phases and may reserve part of the
    step budget for prefills.
    """

    name = "base"
    phase_order: tuple[str, ...] = PHASE_ORDER

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------- interfaces

    def schedule(
        self,
        waiting: Sequence[Request],
        running: Sequence[Request],
        *,
        now: float,
        budget: int,
    ) -> SchedulerOutput:
        work: list[ScheduledWork] = []
        remaining = budget

        for phase in self.phase_order:
            if phase == "decode":
                remaining -= self._schedule_decode(work, running, remaining)
            else:
                remaining -= self._schedule_prefill(work, waiting, running, remaining)

        return SchedulerOutput(work=tuple(work))

    # ---------------------------------------------------------------- phases

    def _schedule_decode(
        self, work: list[ScheduledWork], running: Sequence[Request], budget: int
    ) -> int:
        used = 0
        for request in sorted(running, key=self.decode_order_key):
            if len(work) >= self.config.max_running_requests:
                break
            if request.status is not RequestStatus.DECODING:
                continue
            if used + 1 > budget:
                break
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
        # A request that is mid-prompt stays in the running set, so both sets are
        # candidates: waiting requests can be admitted, running ones need their
        # next chunk. Once a request is active it is no longer in `waiting`, so
        # the two sets never overlap.
        active = {r.id for r in running if r.status.is_active}
        candidates = list(waiting) + list(running)
        occupied = len(active)
        for request in sorted(candidates, key=self.prefill_order_key):
            if request.remaining_prefill == 0:
                continue
            if request.id not in active:
                if occupied >= self.config.max_running_requests:
                    continue
                occupied += 1
            if not self._chunk_fits(request, budget - used):
                continue
            chunk = min(request.remaining_prefill, self._chunk_cap(), budget - used)
            if chunk <= 0:
                continue
            work.append(
                ScheduledWork(request=request, num_new_tokens=chunk, kind=ScheduledKind.PREFILL)
            )
            used += chunk
        return used

    # ------------------------------------------------------------------ hooks

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

    def _chunk_fits(self, request: Request, remaining_budget: int) -> bool:
        """Whether this request can be granted a prefill chunk in this step.

        A chunk is always allowed to be partial, so a prompt larger than the step
        budget is split across steps rather than stalling forever. The chunk cap
        therefore shapes chunks, but never blocks progress.
        """
        return min(self._chunk_cap(), remaining_budget) > 0

    def prefill_order_key(self, request: Request) -> tuple:
        return (request.arrival_time, request.id)

    def decode_order_key(self, request: Request) -> tuple:
        return (request.arrival_time, request.id)


class FCFSPolicy(SchedulerBase):
    """First come, first served, with decode before prefill inside a step."""

    name = "fcfs"


def build_policy(name: str, config: EngineConfig) -> SchedulerBase:
    """Instantiate a scheduling policy by name."""
    policies = {FCFSPolicy.name: FCFSPolicy}
    try:
        policy_cls = policies[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; available: {sorted(policies)}") from None
    return policy_cls(config)
