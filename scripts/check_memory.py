"""Checks for the Phase 2 block-based KV cache manager and admission control.

Run with::

    python scripts/check_memory.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, expect_raises, report

from miniserve import Engine, EngineConfig, Request, RequestStatus, RunnerConfig
from miniserve.memory.block_manager import (
    PagedBlockManager,
    blocks_for_tokens,
)

ZERO_COST = RunnerConfig(
    prefill_base_ms=0.0,
    prefill_ms_per_token=1.0,
    decode_base_ms=0.0,
    decode_ms_per_token=1.0,
    decode_ms_per_context_token=0.0,
)


def make_request(
    prompt_len: int, max_new_tokens: int, arrival: float = 0.0, req_id: str = ""
) -> Request:
    return Request(
        prompt_tokens=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
        arrival_time=arrival,
        id=req_id,
    )


def engine_with(num_blocks: int, block_size: int, budget: int, **overrides) -> Engine:
    settings: dict = {
        "max_batch_tokens": budget,
        "max_running_requests": 8,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "enable_chunked_prefill": True,
        "runner": ZERO_COST,
    }
    settings.update(overrides)
    return Engine(EngineConfig(**settings))


# ------------------------------------------------------------ block arithmetic


def check_blocks_for_tokens_rounds_up() -> None:
    assert blocks_for_tokens(0, 16) == 0
    assert blocks_for_tokens(1, 16) == 1
    assert blocks_for_tokens(16, 16) == 1
    assert blocks_for_tokens(17, 16) == 2
    assert blocks_for_tokens(4096, 16) == 256


def check_paged_block_manager_validates_arguments() -> None:
    expect_raises(ValueError, "num_blocks", lambda: PagedBlockManager(0, 16))
    expect_raises(ValueError, "block_size", lambda: PagedBlockManager(4, 0))


def check_allocation_and_free_conserve_capacity() -> None:
    manager = PagedBlockManager(num_blocks=8, block_size=4)
    requests = [make_request(4 * n, 1) for n in range(1, 4)]

    for request in requests:
        manager.grow_to(request, request.prompt_len)
        request.on_prefill(request.prompt_len)
        manager.assert_invariants()

    assert sum(r.block_table.num_blocks for r in requests) == 1 + 2 + 3
    assert manager.num_free_blocks() == 8 - 6
    assert manager.stats().allocated_blocks == 6

    for request in requests:
        manager.free(request)

    assert manager.num_free_blocks() == 8
    assert manager.num_allocated_blocks == 0
    assert manager.utilization() == 0.0
    manager.assert_invariants()


def check_blocks_are_never_owned_twice() -> None:
    manager = PagedBlockManager(num_blocks=4, block_size=8)
    a = make_request(16, 1, req_id="a")
    b = make_request(16, 1, req_id="b")

    manager.grow_to(a, 16)
    a.on_prefill(16)
    manager.grow_to(b, 16)
    b.on_prefill(16)

    assert set(a.block_table.physical_blocks).isdisjoint(b.block_table.physical_blocks)
    assert manager.num_free_blocks() == 0
    manager.assert_invariants()


def check_exhaustion_raises_rather_than_overcommitting() -> None:
    manager = PagedBlockManager(num_blocks=2, block_size=4)
    request = make_request(64, 1)

    expect_raises(MemoryError, "only 2 are free", lambda: manager.grow_to(request, 64))

    manager.assert_invariants()
    assert manager.num_free_blocks() == 2, "a failed allocation must not leak or reserve"


def check_freeing_an_unowned_block_is_rejected() -> None:
    manager = PagedBlockManager(num_blocks=4, block_size=4)
    request = make_request(4, 1)
    manager.grow_to(request, 4)
    request.on_prefill(4)

    other = make_request(4, 1, req_id="other")
    other.block_table.append(request.block_table[0])

    expect_raises(ValueError, "not owned by other", lambda: manager.free(other))


def check_partial_tail_block_is_reused_before_a_new_block() -> None:
    manager = PagedBlockManager(num_blocks=4, block_size=8)
    request = make_request(20, 1)

    manager.grow_to(request, 6)
    request.on_prefill(6)
    assert manager.num_free_blocks() == 3
    assert manager.num_free_tokens() == 24

    manager.grow_to(request, 8)
    request.on_prefill(2)
    assert manager.num_allocated_blocks == 1, "tokens 7 and 8 fit the same block"

    manager.grow_to(request, 9)
    request.on_prefill(1)
    assert manager.num_allocated_blocks == 2, "token 9 forces a second block"
    manager.assert_invariants()


def check_stats_report_fragmentation() -> None:
    manager = PagedBlockManager(num_blocks=4, block_size=16)
    request = make_request(4, 1)
    manager.grow_to(request, 4)
    request.on_prefill(4)

    stats = manager.stats()

    assert stats.used_tokens == 4
    assert stats.allocated_tokens == 16
    assert stats.utilization == 0.25
    assert stats.internal_fragmentation == 0.75, "12 of 16 reserved tokens are unused"


# ------------------------------------------------------------ engine integration


def check_engine_allocates_blocks_that_match_each_sequence() -> None:
    engine = engine_with(num_blocks=16, block_size=8, budget=32)
    request = engine.submit(make_request(20, 6))

    for _step in engine.run():
        assert engine.memory is not None
        engine.memory.assert_invariants()
        if not request.is_finished:
            # A request that finishes in this step is released in the same step,
            # so its table is empty by the time the step is observed.
            expected = blocks_for_tokens(request.num_tokens, 8)
            assert request.block_table.num_blocks == expected

    memory = engine.memory
    assert memory is not None
    assert memory.stats().used_tokens == 0, "finished requests must release their KV"
    assert memory.num_free_blocks() == 16


def check_kv_limits_prefill_chunk_size() -> None:
    engine = engine_with(num_blocks=2, block_size=8, budget=64)
    engine.submit(make_request(100, 1))

    first = engine.step()

    assert first.num_prefill_tokens == 16, "only 2 blocks of 8 tokens are available"
    assert engine.memory is not None
    assert engine.memory.num_free_blocks() == 0


def check_chunk_size_is_limited_by_free_blocks() -> None:
    """The chunk cap is an upper bound; memory can only lower it."""
    memory = PagedBlockManager(num_blocks=4, block_size=8)
    request = make_request(100, 1)

    # 4 free blocks cover 32 tokens, which is below the 64-token cap.
    assert memory.max_prefill_chunk(request, chunk_cap=64) == 32
    assert memory.max_prefill_chunk(request, chunk_cap=16) == 16, "the cap is still an upper bound"

    memory.grow_to(request, 16)
    request.on_prefill(16)
    assert memory.num_free_blocks() == 2
    assert memory.max_prefill_chunk(request, chunk_cap=64) == 16, (
        "free blocks now only extend coverage by 16 more tokens"
    )


def check_admission_control_waits_instead_of_overcommitting() -> None:
    memory = PagedBlockManager(num_blocks=4, block_size=8)
    request = make_request(40, 1)

    assert memory.blocks_needed(request, 40) == 5
    assert memory.blocks_needed(request, 40) > memory.num_free_blocks()
    assert not memory.can_fit(request, 40), "a request needing 5 of 4 blocks cannot fit"

    # A request that outgrows the pool mid-flight cannot be given another token.
    long_request = make_request(16, 16)
    memory.grow_to(long_request, 16)
    long_request.on_prefill(16)
    assert memory.num_free_blocks() == 2
    for _ in range(16):
        long_request.on_decode(token=0, now=0.0)
        memory.grow_to(long_request, long_request.num_computed_tokens)
    assert memory.num_free_blocks() == 0, "the pool is now full"
    assert memory.blocks_needed(long_request, 1) == 5
    assert not memory.can_fit(long_request, 1)


def check_unfittable_request_stalls_cleanly_instead_of_corrupting_kv() -> None:
    """A prompt too large for the pool stalls, and reports that clearly."""
    engine = engine_with(num_blocks=1, block_size=8, budget=8)
    request = engine.submit(make_request(64, 1))
    memory = engine.memory
    assert memory is not None

    steps = engine.run_to_completion(max_steps=50)

    memory.assert_invariants()
    assert request.status is not RequestStatus.FINISHED, "it cannot have completed"
    assert request.num_computed_tokens == 8, "it cached what the pool could hold, then stopped"
    assert steps[-1].is_stalled, "the engine reports the stall rather than finishing"
    assert memory.num_free_blocks() == 0


def check_finished_requests_return_their_kv_for_the_next_one() -> None:
    """KV reclaimed on finish must be immediately usable by the next request."""
    # One 16-token block holds an 8-token prompt plus 8 output tokens, so a single
    # block must serve both requests in turn.
    engine = engine_with(num_blocks=1, block_size=16, budget=16)
    first = engine.submit(make_request(8, 8, req_id="first"))
    second = engine.submit(make_request(8, 1, req_id="second"))

    engine.run_to_completion()

    assert first.status is RequestStatus.FINISHED
    assert second.status is RequestStatus.FINISHED, (
        "the second request only fits if the first released its block"
    )
    memory = engine.memory
    assert memory is not None
    assert memory.num_free_blocks() == 1


def check_many_requests_share_a_small_pool_without_leaking() -> None:
    engine = engine_with(num_blocks=8, block_size=8, budget=16, max_running_requests=4)
    for i in range(12):
        engine.submit(make_request(5 + i, 3, req_id=f"r{i}"))

    for step in engine.run():
        assert engine.memory is not None
        engine.memory.assert_invariants()
        assert step.free_blocks is not None and step.free_blocks >= 0

    memory = engine.memory
    assert memory is not None
    assert len(engine.finished) == 12, "every request eventually completes"
    assert memory.num_free_blocks() == 8, "all KV returned to the pool"
    assert memory.stats().used_tokens == 0


def check_randomized_workload_holds_all_invariants() -> None:
    """Throw a deterministic random workload at the engine and check the invariants."""
    rng = random.Random(20240922)
    config = EngineConfig(
        max_batch_tokens=64,
        max_running_requests=6,
        # Sized so the theoretical peak demand of 6 concurrently active requests
        # (6 * ceil(80 / 16) = 30 blocks) still fits: without preemption, a pool
        # smaller than the peak can deadlock, which is a separate check.
        num_blocks=64,
        block_size=16,
        enable_chunked_prefill=rng.choice([True, False]),
        max_prefill_chunk=rng.choice([None, 32, 64]),
        runner=ZERO_COST,
    )
    engine = Engine(config)

    clock = 0.0
    for i in range(40):
        clock += rng.uniform(0.0, 0.02)
        engine.submit_at(
            make_request(
                prompt_len=rng.randint(1, 80),
                max_new_tokens=rng.randint(1, 12),
                arrival=clock,
                req_id=f"rand{i}",
            )
        )

    for step in engine.run():
        memory = engine.memory
        assert memory is not None
        memory.assert_invariants()
        assert step.output.num_scheduled_tokens <= config.max_batch_tokens
        assert memory.num_allocated_blocks <= config.num_blocks
        for work in step.output.work:
            request = work.request
            assert request.num_computed_tokens <= request.num_tokens
            assert request.num_generated <= request.max_new_tokens

    memory = engine.memory
    assert memory is not None
    assert len(engine.finished) == 40, "every request must complete"
    assert memory.num_free_blocks() == config.num_blocks


def main() -> int:
    check("blocks_for_tokens rounds up", check_blocks_for_tokens_rounds_up)
    check("block manager argument validation", check_paged_block_manager_validates_arguments)
    check("allocation/free conserve capacity", check_allocation_and_free_conserve_capacity)
    check("blocks are never owned twice", check_blocks_are_never_owned_twice)
    check(
        "exhaustion raises, never overcommits", check_exhaustion_raises_rather_than_overcommitting
    )
    check("freeing an unowned block is rejected", check_freeing_an_unowned_block_is_rejected)
    check("partial tail block is reused", check_partial_tail_block_is_reused_before_a_new_block)
    check("stats report fragmentation", check_stats_report_fragmentation)
    check(
        "engine blocks match each sequence", check_engine_allocates_blocks_that_match_each_sequence
    )
    check("KV limits prefill chunk size", check_kv_limits_prefill_chunk_size)
    check("chunk size is limited by free blocks", check_chunk_size_is_limited_by_free_blocks)
    check(
        "admission control refuses what cannot fit",
        check_admission_control_waits_instead_of_overcommitting,
    )
    check(
        "unfittable request stalls cleanly",
        check_unfittable_request_stalls_cleanly_instead_of_corrupting_kv,
    )
    check(
        "finished requests release KV for the next",
        check_finished_requests_return_their_kv_for_the_next_one,
    )
    check(
        "small pool serves many requests without leaking",
        check_many_requests_share_a_small_pool_without_leaking,
    )
    check("randomized workload holds invariants", check_randomized_workload_holds_all_invariants)
    return report("phase 2 block-based KV manager")


if __name__ == "__main__":
    raise SystemExit(main())
