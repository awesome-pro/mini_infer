"""Step-by-step demo of the engine.

Reproduces the request scenario from the project brief and prints the engine
timeline plus KV pool occupancy, so the scheduler's decisions can be read directly.

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniserve import Engine, EngineConfig, Request, RunnerConfig
from miniserve.visualizations import kv_pool_lines, timeline_lines

PROMPTS = {"R1": 80, "R2": 12, "R3": 300, "R4": 8}
ARRIVALS = {"R1": 0.0, "R2": 0.0, "R3": 1.0, "R4": 3.0}
OUTPUTS = {"R1": 20, "R2": 30, "R3": 10, "R4": 50}


def build_engine() -> Engine:
    return Engine(
        EngineConfig(
            max_batch_tokens=64,
            max_running_requests=4,
            num_blocks=32,
            block_size=16,
            enable_chunked_prefill=True,
            max_prefill_chunk=64,
            runner=RunnerConfig(
                prefill_base_ms=2.0,
                prefill_ms_per_token=0.08,
                decode_base_ms=1.5,
                decode_ms_per_token=0.05,
                decode_ms_per_context_token=0.004,
            ),
        )
    )


def make_requests() -> list[Request]:
    return [
        Request(
            id=name,
            prompt_tokens=list(range(PROMPTS[name])),
            max_new_tokens=OUTPUTS[name],
            arrival_time=ARRIVALS[name],
        )
        for name in ("R1", "R2", "R3", "R4")
    ]


def print_summary(engine: Engine, requests: list[Request]) -> None:
    steps = engine.history
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"engine steps        : {len(steps)}")
    print(f"simulated wall time : {engine.clock.now():.3f} s")
    print(f"prefill tokens      : {sum(s.num_prefill_tokens for s in steps)}")
    print(f"decode tokens       : {sum(s.num_decode_requests for s in steps)}")

    memory = engine.memory
    if memory is not None:
        stats = memory.stats()
        print(
            f"KV pool             : {stats.total_blocks} blocks x "
            f"{stats.block_size} tokens = {stats.total_tokens} token slots"
        )
        print(f"KV peak utilization : {engine.peak_kv_utilization:.1%}")

    print(f"\n{'req':<5}{'prompt':>7}{'out':>5}{'ttft(ms)':>10}{'e2e(ms)':>10}{'status':>10}")
    for req in requests:
        ttft = (
            (req.first_token_time - req.arrival_time) * 1000
            if req.first_token_time is not None
            else float("nan")
        )
        e2e = (
            (req.finish_time - req.arrival_time) * 1000
            if req.finish_time is not None
            else float("nan")
        )
        print(
            f"{req.id:<5}{req.prompt_len:>7}{req.num_generated:>5}"
            f"{ttft:>10.2f}{e2e:>10.2f}{req.status.value:>10}"
        )


def main() -> None:
    engine = build_engine()
    requests = make_requests()

    print("MiniServe demo: token-budget scheduling, chunked prefill, block-based KV")
    config = engine.config
    print(
        f"config: max_batch_tokens={config.max_batch_tokens} "
        f"max_running_requests={config.max_running_requests} "
        f"num_blocks={config.num_blocks} block_size={config.block_size} "
        f"policy={config.policy}"
    )

    engine.run_to_completion(requests)

    print("\n" + "=" * 78)
    print("SCHEDULER TIMELINE   (P<n> = n prefill tokens, D1 = one decode token)")
    print("=" * 78)
    for line in timeline_lines(engine.history, requests):
        print(line)

    print("\n" + "=" * 78)
    print("KV POOL OCCUPANCY   (# = allocated block, . = free block)")
    print("=" * 78)
    for line in kv_pool_lines(engine, samples=10):
        print(line)

    print_summary(engine, requests)


if __name__ == "__main__":
    main()
