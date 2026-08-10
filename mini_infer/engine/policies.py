"""Scheduling policies.

The token budget is always a hard ceiling. Memory is consulted while the plan is
being built: a prefill chunk is clamped to what the KV pool can hold, and a
decode that would need a block the pool does not have is held back for this step.
The remaining policies (prefill-first, decode-first with anti-starvation,
balanced) arrive in a later phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from mini_infer.config import EngineConfig
from mini_infer.engine.request import Request, RequestStatus
from mini_infer.engine.scheduler import ScheduledKind, SchedulerOutput, ScheduledWork
from mini_infer.memory.block_manager import PagedBlockManager

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
        self,
        work: list[ScheduledWork],
        running: Sequence[Request],
        budget: int,
    ) -> int:
        memory = self.memory
        used = 0
        planned_blocks: dict[str, int] = {}

        for request in sorted(running, key=self.decode_order_key):
            if len(work) >= self.config.max_running_requests:
                break
            if request.status is not RequestStatus.DECODING:
                # A request still mid-prompt is served by the prefill phase.
                continue
            if request.is_finished:
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
            if memory is not None and not self._is_final_token(request):
                target = request.sequence_len + 1
                needed = memory.blocks_needed_to_reach(request, target)
                extra = max(0, needed - request.block_table.num_blocks)
                if extra > memory.num_free_blocks() - sum(planned_blocks.values()):
                    # No block for this decode token: claim the slot with zero
                    # tokens and retry once another request releases KV.
                    work.append(
                        ScheduledWork(request=request, num_new_tokens=0, kind=ScheduledKind.DECODE)
                    )
                    continue
                planned_blocks[request.id] = extra
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
        blocks_planned: dict[str, int] = {}

        for request in sorted(candidates, key=self.prefill_order_key):
            if request.remaining_prefill == 0:
                continue
            if request.id not in active:
                if occupied >= self.config.max_running_requests:
                    continue
                if not self._kv_funds_admission(request, blocks_planned):
                    # The pool is completely dry. Activating this request would
                    # race a request that is already running for the last block and
                    # lose, so leave it waiting until KV is reclaimed.
                    continue
                occupied += 1
            chunk = self._chunk_size(request, budget - used, blocks_planned)
            if chunk <= 0:
                # The pool is momentarily full for a request that is already
                # active. Claim the slot with zero tokens so it keeps its place
                # and is retried once other requests release KV.
                work.append(
                    ScheduledWork(request=request, num_new_tokens=0, kind=ScheduledKind.PREFILL)
                )
                continue
            work.append(
                ScheduledWork(request=request, num_new_tokens=chunk, kind=ScheduledKind.PREFILL)
            )
            used += chunk
            if self.memory is not None:
                target = request.sequence_len + chunk
                needed = self.memory.blocks_needed_to_reach(request, target)
                blocks_planned[request.id] = max(
                    0, needed - request.block_table.num_blocks
                )
        return used

    # ------------------------------------------------------------------ hooks

    def _kv_funds_admission(
        self, request: Request, blocks_planned: dict[str, int] | None = None
    ) -> bool:
        """Whether the pool can fund a new request's first prefill chunk.

        Running requests are allowed to be memory-blocked (their work is retried
        and they may release blocks by finishing), but a brand-new request must not
        start when nothing is free: it would compete with already-running work for
        the last block and could only ever lose.
        """
        if self.memory is None:
            return True
        claimed = sum(blocks_planned.values()) if blocks_planned else 0
        return self.memory.num_free_blocks() - claimed > 0

    @staticmethod
    def _is_final_token(request: Request) -> bool:
        """Whether one more token finishes the request.

        The final token's block is released the moment the request completes, so
        refusing it for lack of memory would deadlock a request that is already
        holding most of the pool.
        """
        return request.num_generated + 1 >= request.max_new_tokens

    def _chunk_size(
        self,
        request: Request,
        remaining_budget: int,
        blocks_planned: dict[str, int] | None = None,
    ) -> int:
        """Prefill tokens to grant, clamped by the chunk cap, budget and KV pool."""
        chunk = min(request.remaining_prefill, self._chunk_cap(), remaining_budget)
        if self.memory is not None:
            chunk = min(
                chunk,
                self.memory.max_prefill_chunk(
                    request, blocks_planned=blocks_planned, chunk_cap=chunk
                ),
            )
        return max(0, chunk)

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
