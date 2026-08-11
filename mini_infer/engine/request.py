"""The request model: one generation request and everything the engine tracks about it."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from mini_infer.block_table import BlockTable

_id_counter = itertools.count()


class RequestStatus(str, Enum):
    """Lifecycle of a request.

    ``WAITING -> PREFILLING -> DECODING -> FINISHED``, plus ``PREEMPTED`` for a
    request whose KV blocks were reclaimed and whose prefill must be recomputed.
    """

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    FINISHED = "finished"

    @property
    def is_active(self) -> bool:
        """True while the request occupies a running slot."""
        return self in (RequestStatus.PREFILLING, RequestStatus.DECODING)


@dataclass(slots=True)
class Request:
    """One generation request.

    ``prefilled_tokens`` is the authoritative prefill cursor: it always equals
    the number of prompt tokens whose KV entries exist in the cache, so a
    preempted request simply rewinds it to zero.
    """

    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_time: float
    id: str = ""

    status: RequestStatus = RequestStatus.WAITING
    prefilled_tokens: int = 0
    generated_tokens: list[int] = field(default_factory=list)

    # Physical blocks holding this request's KV, indexed by logical block.
    block_table: BlockTable = field(default_factory=BlockTable)


    first_token_time: float | None = None
    finish_time: float | None = None

    # Observability: engine step on which the request was last admitted.
    admitted_step: int | None = None
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"req-{next(_id_counter)}"
        if self.max_new_tokens < 0:
            raise ValueError(f"{self.id}: max_new_tokens must be non-negative")
        if not self.prompt_tokens:
            raise ValueError(f"{self.id}: prompt_tokens must not be empty")
        if self.prefilled_tokens < 0:
            raise ValueError(f"{self.id}: prefilled_tokens must be non-negative")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def remaining_prefill(self) -> int:
        return self.prompt_len - self.prefilled_tokens

    @property
    def kv_target_tokens(self) -> int:
        """KV slots the request's block table must cover.

        Always the full sequence: a block table covers a prefix of the sequence, so
        the cache grows with the sequence. After preemption the table is empty and
        the recomputation rebuilds that same coverage from the start, which is why
        the requirement never needs to shrink.
        """
        return self.sequence_len

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    @property
    def sequence_len(self) -> int:
        """Tokens whose KV exists: prefilled prompt tokens plus generated ones."""
        return self.prefilled_tokens + len(self.generated_tokens)

    @property
    def is_complete(self) -> bool:
        """True once the output budget is used up."""
        return len(self.generated_tokens) >= self.max_new_tokens

    def context_at_step(self, new_tokens: int) -> int:
        """Sequence length the cache must hold after ``new_tokens`` more tokens."""
        return self.sequence_len + new_tokens

    def on_admitted(self, step: int) -> None:
        self.admitted_step = step

    def on_prefill(self, num_tokens: int) -> None:
        """Advance the prefill cursor.

        Zero tokens is legal: it claims a running slot when the KV pool cannot
        fund any prefill work this step, without pretending tokens were cached.
        """
        if num_tokens < 0:
            raise ValueError(f"{self.id}: prefill chunk must be non-negative")
        if self.prefilled_tokens + num_tokens > self.prompt_len:
            raise ValueError(
                f"{self.id}: prefill of {num_tokens} tokens would exceed prompt "
                f"({self.prefilled_tokens}/{self.prompt_len})"
            )
        self.prefilled_tokens += num_tokens
        if self.prefilled_tokens == self.prompt_len:
            self.status = RequestStatus.DECODING
        elif num_tokens > 0:
            self.status = RequestStatus.PREFILLING

    def on_decode(self, token: int, now: float) -> None:
        """Record one generated token. Returns nothing; callers check stopping."""
        if self.status is not RequestStatus.DECODING:
            raise ValueError(f"{self.id}: cannot decode while {self.status.value}")
        self.generated_tokens.append(token)
        if self.first_token_time is None:
            self.first_token_time = now

    def on_preempted(self) -> None:
        """KV blocks were reclaimed: rewind the prefill cursor so it can be recomputed.

        The output tokens generated before eviction are kept: the recomputation
        prefills the prompt again and re-caches them, so no output is lost.
        """
        self.status = RequestStatus.PREEMPTED
        self.prefilled_tokens = 0
        self.block_table.clear()
        self.num_preemptions += 1

    def on_finished(self, now: float) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_time = now


    def mark_complete(self) -> None:
        """Flag the request as done during scheduling, for prefill-only requests."""
        self.status = RequestStatus.FINISHED
