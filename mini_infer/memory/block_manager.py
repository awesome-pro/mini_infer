"""Fixed-size physical block allocator for KV cache.

Memory is modelled the way paged attention systems do it: a pool of equal-sized
physical blocks, a per-request block table mapping logical positions to physical
blocks, and allocation that grows with the sequence instead of reserving
``prompt + max_new_tokens`` up front.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mini_infer.engine.request import Request
from mini_infer.engine.scheduler import ScheduledWork


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Blocks needed to hold ``num_tokens``, rounding up a partial tail block."""
    if num_tokens <= 0:
        return 0
    return -(-num_tokens // block_size)


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """A snapshot of KV pool pressure."""

    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    sequences: int
    used_tokens: int
    block_size: int

    @property
    def total_tokens(self) -> int:
        return self.total_blocks * self.block_size

    @property
    def allocated_tokens(self) -> int:
        return self.allocated_blocks * self.block_size

    @property
    def utilization(self) -> float:
        """Fraction of the pool handed out to requests."""
        return self.allocated_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def internal_fragmentation(self) -> float:
        """Share of allocated capacity that is reserved but not holding tokens."""
        if self.allocated_tokens == 0:
            return 0.0
        return (self.allocated_tokens - self.used_tokens) / self.allocated_tokens


class BlockPlan:
    """A transactional view of the KV pool for one planning step.

    Holds the real manager plus the blocks already promised to other requests in
    the same step, so every "does this fit?" question has a single answer. Blocks
    are only really taken when the engine allocates, which is what lets the
    scheduler plan speculatively and rebuild the plan after an eviction.
    """

    __slots__ = ("manager", "_claims")

    def __init__(
        self, manager: PagedBlockManager, claims: Mapping[str, int] | None = None
    ) -> None:
        self.manager = manager
        self._claims: dict[str, int] = dict(claims or {})

    @property
    def block_size(self) -> int:
        return self.manager.block_size

    @property
    def num_blocks(self) -> int:
        return self.manager.num_blocks

    def claim(self, request_id: str, blocks: int) -> None:
        self._claims[request_id] = max(0, blocks)

    def claim_of(self, request_id: str) -> int:
        return self._claims.get(request_id, 0)

    def claimed_elsewhere(self, request_id: str | None) -> int:
        return sum(
            blocks for req_id, blocks in self._claims.items() if req_id != request_id
        )

    def free_blocks(self, request_id: str | None = None) -> int:
        """Real free blocks plus what this request claims, minus its rivals' claims."""
        return (
            self.manager.num_free_blocks()
            + self.claim_of(request_id or "")
            - self.claimed_elsewhere(request_id)
        )

    def can_fit(self, request: Request, extra_blocks: int) -> bool:
        """Whether this request may grow by ``extra_blocks`` right now."""
        return extra_blocks <= self.free_blocks(request.id)

    def max_growth_tokens(self, request: Request, cap: int) -> int:
        """Largest number of tokens this request may add, within ``cap``.

        The block table must cover the whole sequence, so the furthest sequence
        length it can reach is the blocks it holds plus the blocks it may still
        take, times the block size. The request's current length is already covered
        by the blocks it holds, so the growth available is the shortfall between
        that reachable length and the current length.
        """
        held = request.block_table.num_blocks + self.claim_of(request.id)
        taken_elsewhere = self.claimed_elsewhere(request.id)
        reachable = (
            held + self.manager.num_free_blocks() - taken_elsewhere
        ) * self.block_size
        return max(0, min(cap, reachable - request.sequence_len))

    def to_dict(self) -> dict[str, int]:
        return dict(self._claims)

    def __repr__(self) -> str:
        return f"BlockPlan(claims={self._claims}, free={self.manager.num_free_blocks()})"


class BlockManager(Protocol):
    """The slice of KV memory the scheduler is allowed to consult."""

    @property
    def block_size(self) -> int: ...

    @property
    def num_blocks(self) -> int: ...

    def num_free_blocks(self) -> int: ...

    def utilization(self) -> float: ...

    def blocks_needed(self, request: Request, num_new_tokens: int) -> int: ...

    def blocks_needed_to_reach(self, request: Request, num_tokens: int) -> int: ...

    def block_plan(self, claims: Mapping[str, int] | None = None) -> BlockPlan:
        """Open a planning view for one scheduling step."""
        return BlockPlan(self, claims)

    def block_plan(self, claims: Mapping[str, int] | None = None) -> BlockPlan:
        """Open a planning view of the pool for one scheduling step."""
        return BlockPlan(self, claims)

    def blocks_needed_for_plan(self, work: Sequence[ScheduledWork]) -> int: ...

    def max_prefill_chunk(
        self,
        request: Request,
        *,
        blocks_planned: Mapping[str, int] | None = None,
        chunk_cap: int | None = None,
    ) -> int: ...


class PagedBlockManager:
    """Allocates and reclaims physical KV blocks.

    Invariants, checked on demand by :meth:`assert_invariants`:

    * one physical block belongs to at most one request
    * a request's block table holds exactly ``ceil(sequence_len / block_size)`` blocks
    * freed blocks return to the pool, so no capacity leaks
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size

        self._free: set[int] = set(range(num_blocks))
        self._owner: dict[int, str] = {}
        self._owned: dict[str, Request] = {}
        self._peak_allocated = 0

    # ------------------------------------------------------------- properties

    @property
    def num_allocated_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_free_tokens(self) -> int:
        """Tokens that could still be cached, counting partial tail blocks."""
        return len(self._free) * self.block_size

    def utilization(self) -> float:
        return self.num_allocated_blocks / self.num_blocks

    @property
    def peak_allocated_blocks(self) -> int:
        return self._peak_allocated

    @property
    def peak_utilization(self) -> float:
        return self._peak_allocated / self.num_blocks

    def owner_of(self, block_id: int) -> str | None:
        return self._owner.get(block_id)

    def tracked_requests(self) -> tuple[Request, ...]:
        return tuple(self._owned.values())

    # --------------------------------------------------------------- planning

    def blocks_needed_to_reach(self, request: Request, num_tokens: int) -> int:
        """Total blocks the request needs to hold ``num_tokens`` tokens."""
        return blocks_for_tokens(num_tokens, self.block_size)

    def blocks_needed(self, request: Request, num_new_tokens: int) -> int:
        """Total blocks needed if ``num_new_tokens`` more tokens are cached."""
        return self.blocks_needed_to_reach(request, request.sequence_len + num_new_tokens)

    def can_fit(self, request: Request, num_new_tokens: int) -> bool:
        extra = self.blocks_needed(request, num_new_tokens) - request.block_table.num_blocks
        return extra <= self.num_free_blocks()

    def block_plan(self, claims: Mapping[str, int] | None = None) -> BlockPlan:
        """Open a planning view for one scheduling step."""
        return BlockPlan(self, claims)

    def block_plan(self, claims: Mapping[str, int] | None = None) -> BlockPlan:
        """Open a planning view of the pool for one scheduling step."""
        return BlockPlan(self, claims)

    def blocks_needed_for_plan(self, work: Sequence[ScheduledWork]) -> int:
        """Blocks required by all scheduled work, counted per request.

        Shrinking a chunk never releases blocks, so a plan that adds ``N`` blocks
        needs ``N`` free blocks even when some of those tokens land in the
        request's partially filled tail block.
        """
        planned_tokens: dict[str, int] = {}
        requests: dict[str, Request] = {}
        for item in work:
            request = item.request
            requests[request.id] = request
            planned_tokens[request.id] = planned_tokens.get(request.id, 0) + item.num_new_tokens

        total = 0
        for request_id, request in requests.items():
            target = request.sequence_len + planned_tokens[request_id]
            needed = self.blocks_needed_to_reach(request, target)
            total += max(0, needed - request.block_table.num_blocks)
        return total

    def max_prefill_chunk(
        self,
        request: Request,
        *,
        blocks_planned: Mapping[str, int] | None = None,
        chunk_cap: int | None = None,
    ) -> int:
        """Largest prefill chunk the pool can hold right now."""
        plan = BlockPlan(self, blocks_planned)
        cap = request.remaining_prefill
        if chunk_cap is not None:
            cap = min(cap, chunk_cap)
        return plan.max_growth_tokens(request, cap)

    # ------------------------------------------------------------- allocation

    def grow_to(self, request: Request, num_tokens: int) -> int:
        """Give the request's block table enough blocks to hold ``num_tokens``.

        Returns the number of physical blocks newly allocated. The caller must
        have established that the blocks are available; over-allocating is never
        allowed, so a request's block table always covers its whole sequence.
        """
        if num_tokens < request.sequence_len:
            raise ValueError(
                f"{request.id}: cannot grow to {num_tokens} tokens, already at "
                f"{request.sequence_len}"
            )
        needed = self.blocks_needed_to_reach(request, num_tokens)
        extra = needed - request.block_table.num_blocks
        if extra <= 0:
            self._owned[request.id] = request
            return 0
        if extra > len(self._free):
            raise MemoryError(
                f"{request.id}: needs {extra} blocks but only {len(self._free)} are free"
            )
        for _ in range(extra):
            block_id = self._free.pop()
            self._owner[block_id] = request.id
            request.block_table.append(block_id)
        self._owned[request.id] = request
        self._peak_allocated = max(self._peak_allocated, self.num_allocated_blocks)
        return extra

    def free(self, request: Request) -> int:
        """Return every block owned by the request to the pool."""
        released = 0
        for block_id in request.block_table.physical_blocks:
            if self._owner.get(block_id) != request.id:
                raise ValueError(
                    f"block {block_id} is not owned by {request.id} "
                    f"(owner={self._owner.get(block_id)!r})"
                )
            del self._owner[block_id]
            self._free.add(block_id)
            released += 1
        request.block_table.clear()
        self._owned.pop(request.id, None)
        return released

    # ------------------------------------------------------------ diagnostics

    def stats(self) -> MemoryStats:
        return MemoryStats(
            total_blocks=self.num_blocks,
            free_blocks=self.num_free_blocks(),
            allocated_blocks=self.num_allocated_blocks,
            sequences=len(self._owned),
            used_tokens=sum(r.sequence_len for r in self._owned.values()),
            block_size=self.block_size,
        )

    def assert_invariants(self) -> None:
        """Fail loudly if the pool's bookkeeping is inconsistent."""
        if len(self._owner) != self.num_blocks - len(self._free):
            raise AssertionError("ownership map and free list disagree")
        if self._free & set(self._owner):
            raise AssertionError("a block is both free and owned")

        for request in self._owned.values():
            expected = blocks_for_tokens(request.sequence_len, self.block_size)
            if request.block_table.num_blocks != expected:
                raise AssertionError(
                    f"{request.id}: holds {request.block_table.num_blocks} blocks but "
                    f"{request.sequence_len} tokens need {expected}"
                )
            for block_id in request.block_table.physical_blocks:
                if self._owner.get(block_id) != request.id:
                    raise AssertionError(f"block {block_id} is not owned by {request.id}")

    def __repr__(self) -> str:
        return (
            f"PagedBlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={self.num_free_blocks()})"
        )
