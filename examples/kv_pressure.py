"""KV pressure demo: show compute scheduling and memory scheduling interacting.

Two long-running requests are admitted into a pool that cannot hold both, so the
engine must decide who keeps memory. Runs the same workload with preemption off
and on, so the difference is visible.

    python examples/kv_pressure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, EngineConfig, Request, RunnerConfig
from mini_infer.engine.engine import StepEventKind
from mini_infer.visualizations import kv_pool_lines, timeline_lines

NUM_BLOCKS = 16
BLOCK_SIZE = 8
PROMPT = 8
OUTPUT = 32


def build_engine(*, preemption: bool) -> Engine:
    return Engine(
        EngineConfig(
            max_batch_tokens=64,
            max_running_requests=4,
            num_blocks=NUM_BLOCKS,
            block_size=BLOCK_SIZE,
            enable_chunked_prefill=True,
            max_prefill_chunk=32,
            enable_preemption=preemption,
            runner=RunnerConfig(
                prefill_base_ms=1.0,
                prefill_ms_per_token=0.1,
                decode_base_ms=1.0,
                decode_ms_per_token=0.1,
                decode_ms_per_context_token=0.0,
            ),
        )
    )


def make_requests() -> list[Request]:
    return [
        Request(
            id=name,
            prompt_tokens=list(range(PROMPT)),
            max_new_tokens=OUTPUT,
            arrival_time=0.0,
        )
        for name in ("A", "B", "C", "D")
    ]


def run_case(preemption: bool) -> None:
    engine = build_engine(preemption=preemption)
    requests = make_requests()
    engine.run_to_completion(requests)

    label = "PREEMPTION ON" if preemption else "PREEMPTION OFF"
    print("\n" + "=" * 78)
    print(f"{label}   (pool = {NUM_BLOCKS} blocks x {BLOCK_SIZE} tokens)")
    print("=" * 78)
    for line in timeline_lines(engine.history, requests, max_rows=26):
        print(line)

    print("\nKV pool over time:")
    for line in kv_pool_lines(engine, samples=8):
        print("  " + line)

    evictions = [e for s in engine.history for e in s.events if e.kind is StepEventKind.PREEMPTED]
    print(f"\n  steps         : {len(engine.history)}")
    print(f"  evictions     : {len(evictions)}")
    for event in evictions[:8]:
        print(f"      {event.request_id} evicted ({event.detail})")
    if len(evictions) > 8:
        print(f"      ... {len(evictions) - 8} more")
    print(f"  recomputations: {sum(r.num_preemptions for r in requests)}")
    for request in requests:
        state = request.status.value
        note = f", recomputed {request.num_preemptions}x" if request.num_preemptions else ""
        print(f"  {request.id}: {state}, {request.num_generated} tokens{note}")
    print(f"  KV left free  : {engine.memory.num_free_blocks()}/{NUM_BLOCKS} blocks")


def main() -> None:
    per_request = -(-(PROMPT + OUTPUT) // BLOCK_SIZE)
    print("MiniServe phase 3 demo - KV pressure, admission control and recompute preemption")
    print(
        f"workload: 4 requests, each {PROMPT} prompt + {OUTPUT} output tokens "
        f"= {PROMPT + OUTPUT} tokens needing {per_request} block(s) each"
    )
    print(
        f"pool: {NUM_BLOCKS} blocks ({NUM_BLOCKS // per_request} requests can be resident "
        f"once fully grown), so the workload cannot fit all at once"
    )
    run_case(preemption=False)
    run_case(preemption=True)
    print(
        "\nRead the two runs together:\n"
        "  * Without preemption, admission control is the only tool available. Each\n"
        "    resident request grows until the pool is dry, then no one can take another\n"
        "    step: every request holds a partial sequence and the engine deadlocks.\n"
        "  * With preemption, the scheduler evicts a victim, hands its blocks to the\n"
        "    request that needs them, and recomputes the victim's prompt later from the\n"
        "    output tokens it kept. The only cost is the recomputed prefill compute.\n"
        "That coupling, not the allocator in isolation, is the point: the scheduler's\n"
        "token decisions and the memory manager's block decisions are made together,\n"
        "which is what lets the engine survive its own memory limit."
    )


if __name__ == "__main__":
    main()
