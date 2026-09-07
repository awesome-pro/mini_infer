"""Observability demo: run a workload and report what it actually did.

Prints the summary shape the README's headline numbers use.

    python examples/metrics_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, EngineConfig, Request, RunnerConfig
from mini_infer.metrics import RunMetrics


def build_requests() -> list[Request]:
    """A mixed workload: short and long prompts, short and long outputs."""
    plan = [
        (12, 24, 0.000),
        (48, 8, 0.002),
        (200, 32, 0.005),
        (8, 64, 0.006),
        (96, 16, 0.011),
        (24, 40, 0.014),
        (320, 12, 0.018),
        (16, 20, 0.020),
    ]
    return [
        Request(
            id=f"R{i}",
            prompt_tokens=list(range(prompt)),
            max_new_tokens=output,
            arrival_time=arrival,
        )
        for i, (prompt, output, arrival) in enumerate(plan)
    ]


def build_engine() -> Engine:
    return Engine(
        EngineConfig(
            max_batch_tokens=64,
            max_running_requests=4,
            num_blocks=48,
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


def print_report(metrics: RunMetrics) -> None:
    print("=" * 62)
    print("WORKLOAD")
    print("=" * 62)
    print(f"  requests            {metrics.num_requests}")
    print(f"  prompt tokens       {metrics.prompt_tokens:,}")
    print(f"  generated tokens    {metrics.generated_tokens:,}")
    print(f"  engine steps        {metrics.num_steps}")
    print(f"  elapsed             {metrics.elapsed_time * 1e3:.1f} ms")

    print("\n" + "=" * 62)
    print("THROUGHPUT")
    print("=" * 62)
    print(f"  output              {metrics.output_throughput:>10,.1f} tok/s")
    print(f"  total               {metrics.total_throughput:>10,.1f} tok/s")
    print(f"  requests            {metrics.requests_per_second:>10,.1f} req/s")

    print("\n" + "=" * 62)
    print("LATENCY (ms)")
    print("=" * 62)
    print(f"  {'':22s}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}")
    for label, mean, p50, p95, p99 in (
        ("TTFT", metrics.mean_ttft, metrics.p50_ttft, metrics.p95_ttft, metrics.p99_ttft),
        ("ITL", metrics.mean_itl, metrics.p50_itl, metrics.p95_itl, metrics.p99_itl),
        ("E2E", metrics.mean_e2e, metrics.p50_e2e, metrics.p95_e2e, float("nan")),
    ):
        cells = "".join(
            f"{value * 1e3:>10.2f}" if value == value else f"{'-':>10}"
            for value in (mean, p50, p95, p99)
        )
        print(f"  {label:22s}{cells}")
    print(
        f"  {'TPOT (per token)':22s}{metrics.mean_tpot * 1e3:>10.2f}"
        f"{metrics.p50_tpot * 1e3:>10.2f}"
          f"{metrics.p95_tpot * 1e3:>10.2f}{'-':>10}")
    print(f"  {'queue time':22s}{metrics.mean_queue_time * 1e3:>10.2f}{'-':>10}"
          f"{metrics.p95_queue_time * 1e3:>10.2f}{'-':>10}")

    print("\n" + "=" * 62)
    print("SCHEDULING AND MEMORY")
    print("=" * 62)
    print(f"  peak KV utilization {metrics.peak_kv_utilization:>10.1%}")
    print(f"  mean KV utilization {metrics.mean_kv_utilization:>10.1%}")
    print(f"  max prefill / step  {metrics.max_prefill_tokens_per_step:>10}")
    print(f"  mean prefill / step {metrics.mean_prefill_tokens_per_step:>10.1f}")
    print(f"  max decode batch    {metrics.max_decode_batch:>10}")
    print(f"  mean decode batch   {metrics.mean_decode_batch:>10.2f}")
    print(f"  preemptions         {metrics.num_preemptions:>10}")


def main() -> None:
    print("MiniServe observability demo - TTFT, ITL, TPOT, throughput, KV pressure")
    engine = build_engine()
    config = engine.config
    print(
        f"config: max_batch_tokens={config.max_batch_tokens} "
        f"max_running_requests={config.max_running_requests} "
        f"num_blocks={config.num_blocks} block_size={config.block_size} "
        f"policy={config.policy}\n"
    )

    requests = build_requests()
    engine.run_to_completion(requests)
    print_report(engine.metrics.summary(elapsed=engine.clock.now()))

    print("\n" + "=" * 62)
    print("PER REQUEST")
    print("=" * 62)
    print(f"  {'req':<5}{'prompt':>7}{'out':>5}{'ttft':>10}{'tpot':>10}{'e2e':>10}{'preempt':>9}")
    for request in requests:
        metrics = engine.metrics.requests[request.id]
        print(
            f"  {request.id:<5}{metrics.prompt_tokens:>7}{metrics.output_tokens:>5}"
            f"{metrics.ttft * 1e3:>10.2f}{metrics.tpot * 1e3:>10.2f}"
            f"{metrics.e2e * 1e3:>10.2f}{metrics.num_preemptions:>9}"
        )


if __name__ == "__main__":
    main()
