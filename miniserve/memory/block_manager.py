"""Fixed-size physical block allocator for KV cache.

Memory is modelled the way paged attention systems do it: a pool of equal-sized
physical blocks, a per-request block table mapping logical positions to physical
blocks, and allocation that grows with the sequence instead of reserving
``prompt + max_new_tokens`` up front.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from miniserve.engine.request import Request
from miniserve.engine.scheduler import ScheduledWork


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """Blocks needed to hold ``num_tokens``, rounding up a partial tail block."""
    if num_tokens <= 0:
        return 0
    return -(-num_tokens // block_size)


def block_hash(tokens: Sequence[int], previous: bytes = b"") -> bytes:
    """Content hash of one full block, chained to the hash of the block before it.

    Chaining is what makes a hit safe. Block *i* can only match another sequence's block
    *i* if every block before it matched too, so a reused block always sits at the same
    absolute positions — and rotary-encoded keys depend on position, so identical keys
    would not otherwise be interchangeable with a later sequence's.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(previous)
    digest.update(array("q", tokens).tobytes())
    return digest.digest()


@dataclass(frozen=True, slots=True)
class MemoryStats:
    """A snapshot of KV pool pressure."""

    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    sequences: int
    used_tokens: int
    block_size: int
    cached_blocks: int = 0
    shared_blocks: int = 0

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

    Holds the real manager plus the blocks already promised to other requests and
    the requests being evicted, all within one step, so every "does this fit?"
    question has a single answer. Nothing here touches the manager: blocks are only
    really taken or released when the engine commits the plan. That is what makes
    the plan speculative and keeps planning separate from committing.
    """

    __slots__ = ("_claims", "_reserved", "_victims", "manager")

    def __init__(
        self,
        manager: PagedBlockManager,
        claims: Mapping[str, int] | None = None,
        victims: Mapping[str, int] | None = None,
        reserved: int = 0,
    ) -> None:
        self.manager = manager
        self._claims: dict[str, int] = dict(claims or {})
        self._victims: dict[str, int] = dict(victims or {})
        #: Distinct blocks this step's cached-prefix attachments take from the pool.
        #: A block shared by several requests is counted once, because it is lost once.
        self._reserved = reserved

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

    @property
    def reserved_blocks(self) -> int:
        """Blocks this step's cached-prefix attachments will take from the pool."""
        return self._reserved

    def claimed_elsewhere(self, request_id: str | None) -> int:
        return sum(
            blocks for req_id, blocks in self._claims.items() if req_id != request_id
        )

    def evict(self, request_id: str, blocks_freed: int) -> None:
        """Mark a request as evicted, releasing its blocks in this plan only."""
        self._victims[request_id] = max(0, blocks_freed)

    def is_evicted(self, request_id: str) -> bool:
        return request_id in self._victims

    @property
    def num_victims(self) -> int:
        return len(self._victims)

    @property
    def blocks_freed(self) -> int:
        return sum(self._victims.values())

    def victims(self) -> dict[str, int]:
        return dict(self._victims)

    def claims(self) -> dict[str, int]:
        """Extra blocks promised to each request by this plan."""
        return {req_id: blocks for req_id, blocks in self._claims.items() if blocks}

    def free_blocks(self, request_id: str | None = None) -> int:
        """Blocks usable right now: real free ones, plus the plan's evictions, plus
        what this request already claims, minus what its rivals claim and minus what
        the step's prefix attachments have been promised."""
        return (
            self.manager.num_free_blocks()
            + self.blocks_freed
            + self.claim_of(request_id or "")
            - self.claimed_elsewhere(request_id)
            - self.reserved_blocks
        )

    def can_fit(self, request: Request, extra_blocks: int) -> bool:
        """Whether this request may grow by ``extra_blocks`` right now."""
        return extra_blocks <= self.free_blocks(request.id)

    def max_growth_tokens(
        self, request: Request, cap: int, *, extra_held: int = 0, cursor: int | None = None
    ) -> int:
        """Largest number of tokens this request may add, within ``cap``.

        The block table must cover everything computed so far, so the furthest
        sequence length reachable is the blocks available to this request times the
        block size. Growth is the shortfall between that and the cursor.

        ``extra_held`` and ``cursor`` let a caller plan for blocks the request is about
        to be given — a cached prefix it will share — without pretending they are already
        in its table.
        """
        # free_blocks already includes this request's own claim, so only the blocks
        # it physically holds need adding.
        held = request.block_table.num_blocks + extra_held
        start = request.num_computed_tokens if cursor is None else cursor
        available = held + self.free_blocks(request.id)
        reachable = available * self.block_size
        return max(0, min(cap, reachable - start))

    def __repr__(self) -> str:
        return (
            f"BlockPlan(claims={self._claims}, victims={self._victims}, "
            f"reserved={self.reserved_blocks}, free={self.manager.num_free_blocks()})"
        )


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

    def block_plan(
        self, claims: Mapping[str, int] | None = None, victims: Mapping[str, int] | None = None
    ) -> BlockPlan:
        """Open a speculative view of the pool for one scheduling step."""
        return BlockPlan(self, claims, victims)

    def blocks_needed_for_plan(self, work: Sequence[ScheduledWork]) -> int: ...

    def max_prefill_chunk(
        self,
        request: Request,
        *,
        blocks_planned: Mapping[str, int] | None = None,
        chunk_cap: int | None = None,
    ) -> int: ...


class PagedBlockManager:
    """Allocates and reclaims physical KV blocks, reusing cached prefixes when asked.

    Invariants, checked on demand by :meth:`assert_invariants`:

    * every block is in exactly one state: free, referenced by one or more requests, or
      cached with no owner
    * a request's block table covers its full sequence length
    * a block is only cached once it is full, and a cached block is never written again
    * freed blocks return to circulation, so no capacity leaks

    With ``enable_prefix_cache`` the manager keeps a content-addressed index of *full*
    blocks, so KV computed once for a token prefix is handed to the next request that
    starts with those same tokens instead of being computed again. Only full blocks are
    cached, and that is what makes sharing safe: a cached block is immutable, and a hit
    always lands on a block boundary, so the block a request writes next is always one
    it owns outright. A cached block nobody holds still counts as free capacity — it is
    evicted on demand, least recently reused first — which keeps every existing
    admission and preemption decision working unchanged.
    """

    def __init__(
        self, num_blocks: int, block_size: int, *, enable_prefix_cache: bool = False
    ) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache

        #: Blocks with no owner and no content: immediately allocatable.
        self._free: set[int] = set(range(num_blocks))
        #: Physical block -> the requests whose block tables reference it.
        self._owners: dict[int, set[str]] = {}
        #: Request id -> request, for every request currently holding blocks.
        self._owned: dict[str, Request] = {}
        #: Full block -> its content hash, and the reverse index, plus reuse order.
        self._hash_of: dict[int, bytes] = {}
        self._cache: dict[bytes, int] = {}
        self._lru: OrderedDict[bytes, None] = OrderedDict()

        self._referenced = 0
        self._cached_free = 0
        self._prefix_hits = 0
        self._prefix_tokens = 0
        self._peak_allocated = 0

    # ------------------------------------------------------------- properties

    @property
    def num_allocated_blocks(self) -> int:
        """Blocks at least one request is holding."""
        return self._referenced

    def num_free_blocks(self) -> int:
        """Blocks a new request could use: idle ones plus evictable cached ones."""
        return len(self._free) + self._cached_free

    def num_free_tokens(self) -> int:
        """Tokens that could still be cached, counting partial tail blocks."""
        return self.num_free_blocks() * self.block_size

    def utilization(self) -> float:
        return self.num_allocated_blocks / self.num_blocks

    @property
    def num_cached_blocks(self) -> int:
        return len(self._cache)

    @property
    def num_shared_blocks(self) -> int:
        return sum(1 for owners in self._owners.values() if len(owners) > 1)

    @property
    def prefix_cache_hits(self) -> int:
        """How many times a request inherited a cached prefix."""
        return self._prefix_hits

    @property
    def prefix_tokens_saved(self) -> int:
        """Tokens whose KV was reused rather than computed."""
        return self._prefix_tokens

    @property
    def peak_allocated_blocks(self) -> int:
        return self._peak_allocated

    @property
    def peak_utilization(self) -> float:
        return self._peak_allocated / self.num_blocks

    def owner_of(self, block_id: int) -> str | None:
        """The single request holding this block, or None if it is free or shared."""
        owners = self._owners.get(block_id)
        if not owners or len(owners) != 1:
            return None
        return next(iter(owners))

    def owners_of(self, block_id: int) -> frozenset[str]:
        return frozenset(self._owners.get(block_id, ()))

    def tracked_requests(self) -> tuple[Request, ...]:
        return tuple(self._owned.values())

    # --------------------------------------------------------------- planning

    def blocks_needed_to_reach(self, request: Request, num_tokens: int) -> int:
        """Total blocks the request needs to hold ``num_tokens`` tokens."""
        return blocks_for_tokens(num_tokens, self.block_size)

    def blocks_needed(self, request: Request, num_new_tokens: int) -> int:
        """Total blocks needed once ``num_new_tokens`` more positions are computed.

        Counted from the compute cursor, like every other memory question: only
        positions that have been computed hold KV.
        """
        return self.blocks_needed_to_reach(
            request, request.num_computed_tokens + num_new_tokens
        )

    def can_fit(self, request: Request, num_new_tokens: int) -> bool:
        extra = self.blocks_needed(request, num_new_tokens) - request.block_table.num_blocks
        return extra <= self.num_free_blocks()

    def block_plan(
        self, claims: Mapping[str, int] | None = None, victims: Mapping[str, int] | None = None
    ) -> BlockPlan:
        """Open a speculative view of the pool for one scheduling step."""
        return BlockPlan(self, claims, victims)

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
            target = request.num_computed_tokens + planned_tokens[request_id]
            needed = self.blocks_needed_to_reach(request, target)
            total += max(0, needed - request.block_table.num_blocks)
        return total

    def can_ever_fit(self, request: Request) -> bool:
        """Whether the request's full sequence could fit an empty pool.

        False for a request whose prompt plus maximum output exceeds the whole pool:
        admitting it can only deadlock the engine, so a benchmark should say so
        rather than report a stall.
        """
        needed = blocks_for_tokens(request.tokens_needed_in_full(), self.block_size)
        return needed <= self.num_blocks

    def max_prefill_chunk(
        self,
        request: Request,
        *,
        blocks_planned: Mapping[str, int] | None = None,
        chunk_cap: int | None = None,
    ) -> int:
        """Largest prefill chunk the pool can hold right now."""
        plan = BlockPlan(self, blocks_planned)
        cap = request.num_uncomputed_tokens
        if chunk_cap is not None:
            cap = min(cap, chunk_cap)
        return plan.max_growth_tokens(request, cap)

    # ------------------------------------------------------------ prefix cache

    def cached_prefix(self, request: Request) -> tuple[int, tuple[int, ...]]:
        """The request's next cached prefix, as ``(position, blocks)``.

        Walks logical blocks forward from the ones the request already holds for as long
        as the sequence's next *full* block hashes to a block in the cache. A query only:
        the caller commits the result with :meth:`attach_prefix`.
        """
        if not self.enable_prefix_cache:
            return (request.num_computed_tokens, ())

        held = request.block_table.num_blocks
        if held:
            previous = self._hash_of.get(request.block_table[held - 1])
            if previous is None:
                # The last block this request holds is not a finished, indexed block, so
                # the chain cannot continue past it.
                return (request.num_computed_tokens, ())
        else:
            previous = b""

        sequence = request.prompt_tokens + request.generated_tokens
        size = self.block_size
        blocks: list[int] = []
        logical = held
        while (logical + 1) * size <= len(sequence):
            digest = block_hash(sequence[logical * size : (logical + 1) * size], previous)
            block = self._cache.get(digest)
            if block is None:
                break
            blocks.append(block)
            previous = digest
            logical += 1
        return (logical * size, tuple(blocks))

    def attach_prefix(self, request: Request, blocks: Sequence[int]) -> int:
        """Share cached blocks into the request's table and count them as reused."""
        size = self.block_size
        for block in blocks:
            self._acquire(block, request.id)
            request.block_table.append(block)
            digest = self._hash_of.get(block)
            if digest is not None:
                self._lru.move_to_end(digest)
        self._owned[request.id] = request
        self._prefix_hits += 1
        self._prefix_tokens += len(blocks) * size
        self._peak_allocated = max(self._peak_allocated, self._referenced)
        return len(blocks)

    def cache_full_blocks(self, request: Request) -> int:
        """Index every full block the request's cursor covers and that is not indexed yet.

        Keyed on the *cursor*, not on an allocation target, and that distinction is a
        correctness one: a block may only be shared once its KV exists. The cursor counts
        positions whose KV has been produced, so indexing here can never publish a block
        the runner has not written yet.
        """
        if not self.enable_prefix_cache:
            return 0
        size = self.block_size
        sequence = request.prompt_tokens + request.generated_tokens
        complete = min(request.num_computed_tokens // size, request.block_table.num_blocks)
        previous = b""
        indexed = 0
        for logical in range(complete):
            block = request.block_table[logical]
            known = self._hash_of.get(block)
            if known is not None:
                previous = known
                continue
            digest = block_hash(sequence[logical * size : (logical + 1) * size], previous)
            previous = digest
            if digest in self._cache:
                # The same content is already indexed under another block; keep the
                # first index and let this block stay private.
                continue
            self._cache[digest] = block
            self._hash_of[block] = digest
            self._lru[digest] = None
            if not self._owners.get(block):
                self._cached_free += 1
            indexed += 1
        return indexed

    def _acquire(self, block: int, request_id: str) -> None:
        """Add one owner to a block, keeping the state counters exact.

        A block arriving from the cache stops being evictable capacity the moment it has
        an owner; a block arriving from the free list was never cached.
        """
        owners = self._owners.setdefault(block, set())
        if not owners:
            self._referenced += 1
            if block in self._hash_of:
                self._cached_free -= 1
        owners.add(request_id)

    def _release(self, block: int, request_id: str) -> None:
        """Drop one owner. The block stays cached if it has content, else it is idle."""
        owners = self._owners.get(block)
        if owners is None or request_id not in owners:
            raise ValueError(
                f"block {block} is not owned by {request_id} "
                f"(owners={sorted(owners) if owners else []})"
            )
        owners.discard(request_id)
        if owners:
            return
        del self._owners[block]
        self._referenced -= 1
        if block in self._hash_of:
            self._cached_free += 1
        else:
            self._free.add(block)

    def _evict_cached(self) -> int | None:
        """Reclaim the least recently reused cached block that nobody is holding."""
        for digest in list(self._lru):
            block = self._cache[digest]
            if self._owners.get(block):
                continue
            del self._lru[digest]
            del self._cache[digest]
            del self._hash_of[block]
            self._cached_free -= 1
            return block
        return None

    # ------------------------------------------------------------- allocation

    def grow_to(self, request: Request, num_tokens: int) -> int:
        """Give the request's block table enough blocks to hold ``num_tokens``.

        Returns the number of physical blocks newly allocated. The caller must
        have established that the blocks are available; over-allocating is never
        allowed, so a request's block table always covers its whole sequence.
        """
        if num_tokens < request.num_computed_tokens:
            raise ValueError(
                f"{request.id}: cannot grow to {num_tokens} tokens, already at "
                f"{request.num_computed_tokens}"
            )
        needed = self.blocks_needed_to_reach(request, num_tokens)
        extra = needed - request.block_table.num_blocks
        if extra <= 0:
            self._owned[request.id] = request
            self.cache_full_blocks(request)
            return 0
        if extra > self.num_free_blocks():
            raise MemoryError(
                f"{request.id}: needs {extra} blocks but only "
                f"{self.num_free_blocks()} are free"
            )
        for _ in range(extra):
            block_id = self._take_block()
            self._acquire(block_id, request.id)
            request.block_table.append(block_id)
        self._owned[request.id] = request
        self.cache_full_blocks(request)
        self._peak_allocated = max(self._peak_allocated, self._referenced)
        return extra

    def _take_block(self) -> int:
        """One block to allocate: an idle one, or the coldest cached one."""
        if self._free:
            return self._free.pop()
        block = self._evict_cached()
        if block is None:  # pragma: no cover - callers check first
            raise MemoryError("the pool has no block left to allocate")
        return block

    def free(self, request: Request) -> int:
        """Release the request's claim on its blocks.

        A block shared with another request survives that request and stays cached; a
        block that was full and is now unreferenced stays in the cache too, available to
        the next request with the same prefix and evictable when capacity is needed.
        """
        self.cache_full_blocks(request)
        released = 0
        for block_id in request.block_table.physical_blocks:
            self._release(block_id, request.id)
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
            used_tokens=sum(r.num_computed_tokens for r in self._owned.values()),
            block_size=self.block_size,
            cached_blocks=self.num_cached_blocks,
            shared_blocks=self.num_shared_blocks,
        )

    def assert_invariants(self) -> None:
        """Fail loudly if the pool's bookkeeping is inconsistent."""
        referenced = {block for block, owners in self._owners.items() if owners}
        if len(referenced) != self._referenced:
            raise AssertionError(
                f"referenced count {self._referenced} but {len(referenced)} blocks are held"
            )
        if referenced & self._free:
            raise AssertionError("a block is both free and held")
        if len(self._free) + self._referenced + self._cached_free != self.num_blocks:
            raise AssertionError(
                f"blocks are unaccounted for: free={len(self._free)} "
                f"held={self._referenced} cached_free={self._cached_free} "
                f"total={self.num_blocks}"
            )
        if len(self._hash_of) != len(self._cache):
            raise AssertionError("the block index and the hash index disagree")
        unreferenced_cached = sum(
            1 for block in self._cache.values() if not self._owners.get(block)
        )
        if unreferenced_cached != self._cached_free:
            raise AssertionError(
                f"{unreferenced_cached} cached blocks have no owner but "
                f"{self._cached_free} are counted"
            )
        for digest, block in self._cache.items():
            if self._hash_of.get(block) != digest:
                raise AssertionError(f"block {block} is indexed inconsistently")
            if block in self._free:
                raise AssertionError(f"cached block {block} is also in the free list")

        for request in self._owned.values():
            expected = blocks_for_tokens(request.num_computed_tokens, self.block_size)
            if request.block_table.num_blocks != expected:
                raise AssertionError(
                    f"{request.id}: holds {request.block_table.num_blocks} blocks but "
                    f"{request.num_computed_tokens} computed tokens need {expected}"
                )
            for block_id in request.block_table.physical_blocks:
                if request.id not in self._owners.get(block_id, ()):
                    raise AssertionError(f"block {block_id} is not held by {request.id}")

    def __repr__(self) -> str:
        return (
            f"PagedBlockManager(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"free={self.num_free_blocks()}, cached={self.num_cached_blocks})"
        )
