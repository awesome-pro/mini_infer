"""Checks for prefix caching: content-addressed blocks, sharing, and reuse.

The features under test are the ones that make shared KV safe rather than merely
convenient: a block is only cached once it is full, so it can never be written again; a
hit always lands on a block boundary, so the block a request writes next is its own; and
a block's references are counted, so freeing one owner cannot pull memory out from under
another.

Run with::

    python scripts/check_prefix_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, report

from mini_infer import Engine, EngineConfig, Request, RunnerConfig
from mini_infer.engine.request import RequestStatus
from mini_infer.memory.block_manager import (
    PagedBlockManager,
    block_hash,
    blocks_for_tokens,
)

ZERO_COST = RunnerConfig(
    prefill_base_ms=0.0,
    prefill_ms_per_token=1.0,
    decode_base_ms=0.0,
    decode_ms_per_token=1.0,
    decode_ms_per_context_token=0.0,
)

SHARED = list(range(12))


def config(**overrides) -> EngineConfig:
    settings: dict = {
        "max_batch_tokens": 64,
        "max_running_requests": 1,
        "num_blocks": 32,
        "block_size": 4,
        "enable_prefix_cache": True,
        "runner": ZERO_COST,
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def request(
    prompt: list[int], *, arrival: float = 0.0, max_new: int = 3, req_id: str = ""
) -> Request:
    return Request(prompt_tokens=prompt, max_new_tokens=max_new, arrival_time=arrival, id=req_id)


def allocate(manager: PagedBlockManager, request_obj: Request, num_tokens: int) -> int:
    """Advance the cursor the way the engine does, then allocate.

    The manager indexes full blocks from the cursor, so a caller that skips the cursor is
    modelling something the engine never does: allocating KV for positions that have not
    been computed.
    """
    request_obj.on_prefill(num_tokens)
    return manager.grow_to(request_obj, num_tokens)


def staggered_run(prefix_cache: bool, tails=((90, 91), (92, 93), (94, 95)), **overrides) -> dict:
    """Three requests sharing a 12-token prefix, arriving one after another."""
    engine = Engine(config(enable_prefix_cache=prefix_cache, **overrides))
    arrivals = []
    clock = 0.0
    for index, tail in enumerate(tails):
        arrivals.append(request(SHARED + list(tail), arrival=clock, req_id=f"r{index}"))
        clock += 0.05
    steps = engine.run_to_completion(arrivals)
    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    return {
        "engine": engine,
        "prefill_tokens": sum(step.num_prefill_tokens for step in steps),
        "hits": metrics.prefix_cache_hits,
        "reused": metrics.cached_prefix_tokens,
        "steps": len(steps),
        "finished": metrics.num_finished,
        "generated": [r.num_generated for r in engine.requests.values()],
    }


# ------------------------------------------------------------------- hashing


def check_block_hash_is_stable_and_chained() -> None:
    assert block_hash([1, 2, 3, 4]) == block_hash([1, 2, 3, 4]), "hashing must be stable"
    assert block_hash([1, 2, 3, 4]) != block_hash([4, 3, 2, 1]), "order must matter"
    assert block_hash([1, 2, 3, 4]) != block_hash([1, 2, 3, 5]), "content must matter"
    assert block_hash([1, 2, 3, 4], b"prefix") != block_hash([1, 2, 3, 4]), (
        "the same tokens after a different prefix must hash differently, or a block "
        "could be reused at the wrong absolute positions"
    )
    assert len({block_hash(list(range(i, i + 4))) for i in range(50)}) == 50


def check_only_full_blocks_are_cached() -> None:
    """A partial tail block is never shared: it is the block a request is writing."""
    engine = Engine(config(block_size=4))
    engine.submit(request(list(range(14)), max_new=1, req_id="r0"))
    engine.run_to_completion()

    memory = engine.memory
    assert memory is not None
    assert memory.num_cached_blocks == 3, "14 tokens fill 3 blocks; the 4th is partial"
    assert memory.num_cached_blocks == 14 // 4


# ------------------------------------------------------------------ sharing


def check_a_second_request_shares_the_same_blocks() -> None:
    """The whole point: the same physical blocks, referenced twice."""
    engine = Engine(config(max_running_requests=2, block_size=4))
    engine.submit(request([*SHARED, 90, 91], max_new=20, req_id="first"))
    engine.step()  # prefill: the shared blocks are now computed and indexed

    second = engine.submit(request([*SHARED, 92, 93], max_new=20, req_id="second"))
    step = engine.step()

    memory = engine.memory
    assert memory is not None
    shared = [
        block
        for block in second.block_table.physical_blocks
        if len(memory.owners_of(block)) == 2
    ]
    assert shared == list(second.block_table.physical_blocks[:3]), (
        f"the first three blocks must be shared, got {second.block_table.physical_blocks}"
    )
    assert memory.num_shared_blocks >= 3
    reused = [e for e in step.events if e.kind.value == "prefix_cached"]
    assert reused and reused[0].num_tokens == 12, "the hit covers the shared prefix"


def check_a_hit_stops_at_the_first_divergence() -> None:
    """A shared first block is reusable; the block after it is not."""
    engine = Engine(config(block_size=4))
    engine.submit(request([1, 2, 3, 4, 9, 9, 9, 9, 9, 9, 9, 9], max_new=1, req_id="r0"))
    engine.run_to_completion()

    engine.submit(request([1, 2, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7], max_new=1, req_id="r1"))
    step = engine.step()

    reused = [e for e in step.events if e.kind.value == "prefix_cached"]
    assert reused and reused[0].num_tokens == 4, "only the identical first block may be reused"
    memory = engine.memory
    assert memory is not None
    assert memory.prefix_tokens_saved == 4


def check_shared_blocks_outlive_one_owner() -> None:
    """Reference counting, tested directly: one owner leaving must not free the block."""
    manager = PagedBlockManager(8, 4, enable_prefix_cache=True)
    prompt = [1, 2, 3, 4] * 2
    original = request(prompt, req_id="original")
    allocate(manager, original, len(prompt))
    manager.free(original)
    assert manager.num_cached_blocks == 2 and manager.num_free_blocks() == 8

    first = request(prompt, req_id="first")
    position, blocks = manager.cached_prefix(first)
    manager.attach_prefix(first, blocks)
    first.on_prefill(position - first.num_computed_tokens)
    assert position == len(prompt)
    assert manager.owners_of(blocks[0]) == frozenset({"first"})
    assert manager.num_free_blocks() == 6, "a held block is not free capacity"

    second = request(prompt, req_id="second")
    position, blocks = manager.cached_prefix(second)
    manager.attach_prefix(second, blocks)
    second.on_prefill(position - second.num_computed_tokens)
    manager.assert_invariants()
    assert manager.owners_of(blocks[0]) == frozenset({"first", "second"})
    assert manager.num_shared_blocks == 2

    manager.free(first)
    manager.assert_invariants()
    assert manager.owners_of(blocks[0]) == frozenset({"second"}), "the survivor still holds it"
    assert manager.num_free_blocks() == 6

    manager.free(second)
    manager.assert_invariants()
    assert manager.owners_of(blocks[0]) == frozenset(), "nobody holds it now"
    assert manager.num_cached_blocks == 2, "but it stays cached for the next request"
    assert manager.num_free_blocks() == 8, "cached blocks count as free capacity"


def check_cached_blocks_are_evicted_before_capacity_runs_out() -> None:
    """A cached block is free capacity: the pool evicts the coldest one on demand."""
    manager = PagedBlockManager(4, 4, enable_prefix_cache=True)
    filler = request([1, 2, 3, 4] * 4, req_id="filler")
    allocate(manager, filler, 16)
    manager.free(filler)
    assert manager.num_cached_blocks == 4
    assert manager.num_free_blocks() == 4, "every block is cached, none is idle"

    newcomer = request([9, 9, 9, 9], req_id="new")
    allocate(manager, newcomer, 4)

    assert manager.num_free_blocks() == 3, "one cached block became the newcomer's own"
    assert manager.num_cached_blocks == 4, "three survivors plus the newcomer's block"
    manager.assert_invariants()


def check_least_recently_used_block_goes_first() -> None:
    manager = PagedBlockManager(2, 4, enable_prefix_cache=True)
    first = request([1, 1, 1, 1], req_id="first")
    allocate(manager, first, 4)
    manager.free(first)
    second = request([2, 2, 2, 2], req_id="second")
    allocate(manager, second, 4)
    manager.free(second)

    # Touch the first block again, so the second one is now colder.
    reuse = request([1, 1, 1, 1], req_id="reuse")
    position, blocks = manager.cached_prefix(reuse)
    assert position == 4 and len(blocks) == 1
    manager.attach_prefix(reuse, blocks)
    manager.free(reuse)

    newcomer = request([5, 5, 5, 5], req_id="new")
    allocate(manager, newcomer, 4)
    manager.assert_invariants()
    kept = manager._cache
    assert block_hash([1, 1, 1, 1]) in kept, "the recently reused block survived"
    assert block_hash([2, 2, 2, 2]) not in kept, "the colder entry was evicted first"


# --------------------------------------------------------------- integration


def check_reuse_cuts_prefill_work_and_keeps_output() -> None:
    off = staggered_run(prefix_cache=False)
    on = staggered_run(prefix_cache=True)

    assert off["hits"] == 0 and off["reused"] == 0
    assert off["prefill_tokens"] == 3 * len([*SHARED, 0, 0]), "each request computes its own"
    assert on["hits"] == 2, f"the second and third requests should hit, got {on['hits']}"
    assert on["reused"] == 2 * len(SHARED)
    assert on["prefill_tokens"] < off["prefill_tokens"] * 0.5, (
        f"reuse should remove most prefill work: {on['prefill_tokens']} vs {off['prefill_tokens']}"
    )
    assert on["generated"] == off["generated"], "the cache must not change what is generated"
    assert on["finished"] == off["finished"] == 3


def check_reuse_survives_a_finished_run() -> None:
    """Blocks stay reusable after every owner is gone, which is the common case."""
    engine = Engine(config())
    engine.submit(request([*SHARED, 90, 91], max_new=1, req_id="r0"))
    engine.run_to_completion()
    assert engine.memory is not None
    assert engine.memory.num_cached_blocks == 3

    engine.submit(request([*SHARED, 92, 93], max_new=1, req_id="r1"))
    engine.run_to_completion()

    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    assert metrics.cached_prefix_tokens == len(SHARED)
    assert engine.requests["r1"].num_generated == 1, "the reused prefix still produces output"


def check_a_recomputing_request_reuses_its_own_prefix() -> None:
    """Preemption rewinds the cursor; the prefix is still cached, so it comes back free."""
    engine = Engine(
        config(num_blocks=9, block_size=4, max_running_requests=3, max_batch_tokens=8)
    )
    engine.submit(request([*SHARED, 90, 91], max_new=16, req_id="heavy"))
    engine.submit(request([*SHARED, 92, 93], max_new=16, req_id="other"))

    for step in engine.run():
        memory = engine.memory
        assert memory is not None
        memory.assert_invariants()
        assert step.output.num_scheduled_tokens <= 8

    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    memory = engine.memory
    assert memory is not None
    assert metrics.num_finished == 2, "both requests must finish"
    assert metrics.num_preemptions > 0, "this pool is tight enough to force a recompute"
    assert metrics.cached_prefix_tokens > 0, "the recompute should have reused the prefix"
    assert memory.num_free_blocks() == memory.num_blocks


def check_prefix_cache_off_is_inert() -> None:
    engine = Engine(config(enable_prefix_cache=False))
    assert engine.memory is not None
    assert engine.memory.enable_prefix_cache is False
    engine.submit(request([*SHARED, 90, 91], max_new=1, req_id="r0"))
    engine.run_to_completion()
    engine.submit(request([*SHARED, 90, 91], max_new=1, req_id="r1"))
    engine.run_to_completion()

    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    assert engine.memory.num_cached_blocks == 0
    assert metrics.prefix_cache_hits == 0 and metrics.cached_prefix_tokens == 0


def check_shared_blocks_survive_a_real_engine_run() -> None:
    """Invariants hold on every step of a burst that keeps hitting the same prefix."""
    engine = Engine(config(max_running_requests=3, block_size=4, num_blocks=24))
    for index in range(12):
        engine.submit(request([*SHARED, 90, 91 + index], arrival=0.05 * index, req_id=f"r{index}"))

    for _step in engine.run():
        memory = engine.memory
        assert memory is not None
        memory.assert_invariants()
        for request_obj in engine.requests.values():
            if request_obj.is_finished:
                assert request_obj.block_table.num_blocks == 0

    memory = engine.memory
    assert memory is not None
    memory.assert_invariants()
    assert memory.num_free_blocks() == memory.num_blocks, "all capacity is accounted for"
    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    assert metrics.num_finished == 12
    assert metrics.prefix_cache_hits >= 8, (
        f"expected most requests to hit: {metrics.prefix_cache_hits}"
    )
    for request_obj in engine.requests.values():
        assert request_obj.status is RequestStatus.FINISHED
        assert request_obj.num_generated == 3


def check_metrics_count_the_reused_tokens() -> None:
    on = staggered_run(prefix_cache=True)
    engine = on["engine"]
    assert engine.memory is not None
    assert engine.memory.prefix_tokens_saved == on["reused"]
    assert engine.memory.prefix_cache_hits == on["hits"]
    # Blocks are four tokens wide, and every hit is a whole number of blocks.
    assert on["reused"] % engine.config.block_size == 0


def check_block_size_one_still_reuses() -> None:
    """The degenerate layout: every token is its own block."""
    engine = Engine(config(block_size=1, num_blocks=64))
    engine.submit(request([5] * 8, max_new=1, req_id="r0"))
    engine.run_to_completion()
    engine.submit(request([5] * 8, max_new=1, req_id="r1"))
    engine.run_to_completion()

    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    assert metrics.cached_prefix_tokens == 8, (
        f"expected a full hit, got {metrics.cached_prefix_tokens}"
    )


def check_a_prompt_shorter_than_a_block_is_never_shared() -> None:
    engine = Engine(config(block_size=16, num_blocks=8))
    engine.submit(request([1, 2, 3], max_new=2, req_id="r0"))
    engine.run_to_completion()
    engine.submit(request([1, 2, 3], max_new=2, req_id="r1"))
    engine.run_to_completion()

    assert engine.memory is not None
    assert blocks_for_tokens(3, 16) == 1, "the prompt occupies a single, partial block"
    assert engine.memory.num_cached_blocks == 0, "a partial block is never shared"
    assert engine.metrics.summary(elapsed=engine.clock.now()).cached_prefix_tokens == 0


def main() -> int:
    check("block hash is stable and chained", check_block_hash_is_stable_and_chained)
    check("only full blocks are cached", check_only_full_blocks_are_cached)
    check("a second request shares the same blocks", check_a_second_request_shares_the_same_blocks)
    check("a hit stops at the first divergence", check_a_hit_stops_at_the_first_divergence)
    check("shared blocks outlive one owner", check_shared_blocks_outlive_one_owner)
    check(
        "cached blocks are evicted before capacity runs out",
        check_cached_blocks_are_evicted_before_capacity_runs_out,
    )
    check("least recently used block goes first", check_least_recently_used_block_goes_first)
    check(
        "reuse cuts prefill work and keeps output",
        check_reuse_cuts_prefill_work_and_keeps_output,
    )
    check("reuse survives a finished run", check_reuse_survives_a_finished_run)
    check(
        "a recomputing request reuses its own prefix",
        check_a_recomputing_request_reuses_its_own_prefix,
    )
    check("the cache is inert when disabled", check_prefix_cache_off_is_inert)
    check("shared blocks survive a real engine run", check_shared_blocks_survive_a_real_engine_run)
    check("metrics count the reused tokens", check_metrics_count_the_reused_tokens)
    check("block_size 1 still reuses", check_block_size_one_still_reuses)
    check(
        "a prompt shorter than a block is never shared",
        check_a_prompt_shorter_than_a_block_is_never_shared,
    )
    return report("prefix caching")


if __name__ == "__main__":
    raise SystemExit(main())
