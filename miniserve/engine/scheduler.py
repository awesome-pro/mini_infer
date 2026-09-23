"""The scheduling contract.

A scheduler never says "prefill r3". It says "give r3 thirty compute tokens".
That generality is what lets the same contract express full prefills, chunked
prefills and single-token decodes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from miniserve.engine.request import Request


class ScheduledKind(StrEnum):
    """What kind of work a scheduled token count represents."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class ScheduledWork:
    """``num_new_tokens`` compute slots granted to one request this step."""

    request: Request
    num_new_tokens: int
    kind: ScheduledKind

    @property
    def request_id(self) -> str:
        return self.request.id

    def __post_init__(self) -> None:
        if self.num_new_tokens < 0:
            raise ValueError(
                f"{self.request_id}: scheduled tokens must be non-negative, "
                f"got {self.num_new_tokens}"
            )
        if self.kind is ScheduledKind.DECODE and self.num_new_tokens > 1:
            raise ValueError(
                f"{self.request_id}: a decode step produces at most one token, "
                f"got {self.num_new_tokens}"
            )
        if self.kind is ScheduledKind.PREFILL:
            uncomputed = self.request.num_uncomputed_tokens
            if self.num_new_tokens > uncomputed:
                raise ValueError(
                    f"{self.request_id}: prefill chunk {self.num_new_tokens} exceeds "
                    f"the {uncomputed} uncomputed positions"
                )


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Everything that executes in one engine step, plus any KV evictions it needs."""

    work: Sequence[ScheduledWork]
    #: Victim request id -> the request whose work required the eviction. Planning
    #: only decides this; the engine releases the blocks and rewinds the victim.
    preemptions: dict[str, str] = field(default_factory=dict)
    #: Extra physical blocks each scheduled request needs for this step's work,
    #: computed while planning. The engine commits exactly this.
    allocations: dict[str, int] = field(default_factory=dict)
    #: Cached prefix blocks to share into the scheduled requests before they run, keyed
    #: by request. Planning decided these; the engine refcounts them.
    attachments: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def num_preemptions(self) -> int:
        return len(self.preemptions)

    @property
    def num_blocks_allocated(self) -> int:
        return sum(self.allocations.values())

    @property
    def num_blocks_attached(self) -> int:
        return sum(len(blocks) for blocks in self.attachments.values())

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(w.num_new_tokens for w in self.work)

    @property
    def prefill_work(self) -> tuple[ScheduledWork, ...]:
        return tuple(w for w in self.work if w.kind is ScheduledKind.PREFILL)

    @property
    def decode_work(self) -> tuple[ScheduledWork, ...]:
        return tuple(w for w in self.work if w.kind is ScheduledKind.DECODE)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(w.num_new_tokens for w in self.prefill_work)

    @property
    def num_decode_requests(self) -> int:
        return len(self.decode_work)

    @property
    def is_empty(self) -> bool:
        return not self.work

    def __len__(self) -> int:
        return len(self.work)

    def __iter__(self) -> Iterator[ScheduledWork]:
        return iter(self.work)

    def __contains__(self, request_id: object) -> bool:
        return any(w.request_id == request_id for w in self.work)

    def tokens_for(self, request_id: str) -> int:
        for work in self.work:
            if work.request_id == request_id:
                return work.num_new_tokens
        return 0
