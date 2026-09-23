"""The request model: one generation request and everything the engine tracks about it."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import StrEnum

from miniserve.block_table import BlockTable

_id_counter = itertools.count()
#: Submission order, used wherever 'younger' has to mean something when two
#: requests arrived at the same instant: a burst has no arrival-time order at all.
_sequence_counter = itertools.count()


class RequestStatus(StrEnum):
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

    The sequence is ``prompt_tokens + generated_tokens`` and grows at the tail.
    ``num_computed_tokens`` is the single cursor over that sequence: it counts how
    many leading positions have KV entries in the cache. Prefill advances it over
    the prompt, decode extends the sequence and advances it by one, and preemption
    rewinds it to zero so the sequence is recomputed from the start. There is no
    separate prefill cursor, because the only question the engine ever asks is how
    much of the sequence is cached.
    """

    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_time: float
    id: str = ""

    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # Physical blocks holding this request's KV, indexed by logical block.
    block_table: BlockTable = field(default_factory=BlockTable)

    first_token_time: float | None = None
    finish_time: float | None = None

    #: Monotonic submission order. Arrival times tie constantly — every burst request
    #: arrives at the same instant — so age needs a tie-break that is not the id string.
    sequence: int = field(default_factory=lambda: next(_sequence_counter))

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
        if self.num_computed_tokens < 0:
            raise ValueError(f"{self.id}: num_computed_tokens must be non-negative")

    # ------------------------------------------------------------- geometry

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.generated_tokens)

    @property
    def num_tokens(self) -> int:
        """Total sequence length: everything the request will ever attend over so far."""
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def num_uncomputed_tokens(self) -> int:
        """Positions after the cursor that still need a forward pass.

        Normally this is the rest of the prompt. After preemption it also covers the
        output tokens generated before eviction, whose KV the recomputation rebuilds.
        """
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill_complete(self) -> bool:
        """True once the whole prompt has KV, so decode may begin."""
        return self.num_computed_tokens >= self.prompt_len

    @property
    def num_uncomputed_prompt_tokens(self) -> int:
        """Prompt positions still to be computed, ignoring generated ones."""
        return max(0, self.prompt_len - self.num_computed_tokens)

    @property
    def is_complete(self) -> bool:
        """True once the output budget is used up."""
        return len(self.generated_tokens) >= self.max_new_tokens

    @property
    def is_finished(self) -> bool:
        """True once the request has been finalised by the engine."""
        return self.status is RequestStatus.FINISHED

    def tokens_needed_in_full(self) -> int:
        """Sequence length the request will hold once fully generated.

        Used to detect a request that can never fit the KV pool, which is a
        configuration limit rather than a scheduling problem: no policy can admit it
        because one request alone would need more memory than the pool has.
        """
        return self.num_tokens + (self.max_new_tokens - self.num_generated)

    def context_at_step(self, new_tokens: int) -> int:
        """Sequence length the cache must hold after ``new_tokens`` more tokens."""
        return self.num_computed_tokens + new_tokens

    # ----------------------------------------------------------- transitions

    def on_admitted(self, step: int) -> None:
        self.admitted_step = step

    def on_prefill(self, num_tokens: int) -> None:
        """Advance the cursor over the sequence.

        Zero tokens is legal: it claims a running slot when the KV pool cannot fund
        any work this step, without pretending a position was computed.
        """
        if num_tokens < 0:
            raise ValueError(f"{self.id}: prefill chunk must be non-negative")
        if self.num_computed_tokens + num_tokens > self.num_tokens:
            raise ValueError(
                f"{self.id}: computing {num_tokens} tokens would pass the sequence "
                f"end ({self.num_computed_tokens}/{self.num_tokens})"
            )
        self.num_computed_tokens += num_tokens
        if num_tokens > 0:
            self.status = (
                RequestStatus.DECODING
                if self.is_prefill_complete
                else RequestStatus.PREFILLING
            )

    def on_decode(self, token: int, now: float) -> None:
        """Append one generated token and count its position as computed."""
        if self.status is not RequestStatus.DECODING:
            raise ValueError(f"{self.id}: cannot decode while {self.status.value}")
        self.generated_tokens.append(token)
        self.num_computed_tokens += 1
        if self.first_token_time is None:
            self.first_token_time = now

    def on_sampled(self, token: int, now: float) -> None:
        """Append a token a real forward pass sampled, leaving the cursor behind it.

        A runner with real logits samples the next token while computing the positions
        it was granted, so the token it produces has no KV yet: the sequence grows by
        one while the cursor stays where it is, and the *next* step computes that
        position. The result is one pending position per decoding request, which is
        what the decode phase is for.

        The simulated runner has nothing to sample, so it appends on the decode step
        instead (:meth:`on_decode`) and keeps the cursor and the sequence in lockstep.
        """
        self.generated_tokens.append(token)
        if self.first_token_time is None:
            self.first_token_time = now

    def on_preempted(self) -> None:
        """KV blocks were reclaimed: rewind the cursor so the sequence is recomputed.

        The generated tokens are kept: recomputation walks the sequence again and
        restores their KV, so no output is lost.
        """
        self.status = RequestStatus.PREEMPTED
        self.num_computed_tokens = 0
        self.block_table.clear()
        self.num_preemptions += 1

    def on_finished(self, now: float) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_time = now

    def mark_complete(self) -> None:
        """Flag the request as done during scheduling, for prefill-only requests."""
        self.status = RequestStatus.FINISHED
