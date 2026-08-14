"""Benchmark demo: sweep the axes the experiments care about.

    python examples/benchmark_demo.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import EngineConfig, RunnerConfig  # noqa: E402
from mini_infer.benchmark import (  # noqa: E402
    Report,
    WorkloadSpec,
    output_profile,
    prompt_profile,
    run_case,
)
from mini_infer.visualizations import metric_panel, trend_summary  # noqa: E402

RUNNER = dict(
    prefill_base_ms=2.0,
    prefill_ms_per_token=0.08,
    decode_base_ms=1.5,
    decode_ms_per_token=0.05,
    decode_ms_per_context_token=0.004,
)


def base_workload(requests: int = 120, seed: int = 4) -> WorkloadSpec:
    return WorkloadSpec(
        num_requests=requests,
        prompt=prompt_profile("chat"),
        output=output_profile("balanced"),
        arrival="poisson",
        rate=16.0,
        seed=seed,
    )


def _config(**overrides) -> EngineConfig:
    settings: dict = {
        "max_batch_tokens": 64,
        "max_running_requests": 4,
        "num_blocks": 192,
        "block_size": 16,
        "runner": RunnerConfig(**RUNNER),
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def sweep_rate(workload: WorkloadSpec) -> None:
    rates = [2, 4, 8, 16, 32, 64, 128, 256]
    results = [run_case(replace(workload, rate=rate), _config()) for rate in rates]
    labels = [f"{rate}/s" for rate in rates]

    print("=" * 74)
    print("SWEEP 1: arrival rate (open-loop poisson)")
    print("=" * 74)
    print(
        metric_panel(
            labels,
            {
                "throughput (tok/s)": [r.metrics.output_throughput for r in results],
                "TTFT p95 (ms)": [r.metrics.p95_ttft * 1e3 for r in results],
                "queue p95 (ms)": [r.metrics.p95_queue_time * 1e3 for r in results],
            },
        )
    )
    print()
    print(
        trend_summary(
            labels,
            {
                "throughput": [r.metrics.output_throughput for r in results],
                "TTFT p95": [r.metrics.p95_ttft * 1e3 for r in results],
            },
        )
    )
    saturated = [r for r in results if r.metrics.p95_queue_time > 0.05]
    if saturated:
        print(
            f"\n  Past around {saturated[0].workload.spec.rate:g} req/s the engine saturates:\n"
            "  throughput flattens while queueing delay grows, so extra offered load\n"
            "  becomes latency rather than work done. That is the capacity limit."
        )


def sweep_kv(workload: WorkloadSpec) -> None:
    pools = [48, 96, 192, 384]
    results = [
        run_case(workload, _config(num_blocks=blocks, block_size=16)) for blocks in pools
    ]
    print("\n" + "=" * 74)
    print("SWEEP 2: KV pool size (fixed rate, tight pools force preemption)")
    print("=" * 74)
    print(
        Report().render(
            [
                {
                    "label": f"{blocks} blk",
                    **r.metrics.as_row(),
                }
                for blocks, r in zip(pools, results, strict=True)
            ],
            title="",
        )
    )
    print()
    print(
        trend_summary(
            [f"{b}" for b in pools],
            {
                "evictions": [r.metrics.num_preemptions for r in results],
                "peak KV %": [r.metrics.peak_kv_utilization * 100 for r in results],
                "throughput": [r.metrics.output_throughput for r in results],
            },
        )
    )


def sweep_burst(workload: WorkloadSpec) -> None:
    print("\n" + "=" * 74)
    print("SWEEP 3: arrival process at the same offered load")
    print("=" * 74)
    modes = [
        ("burst", replace(workload, arrival="burst")),
        ("poisson", replace(workload, arrival="poisson", rate=16.0)),
        ("closed_loop", replace(workload, arrival="closed_loop", concurrency=4)),
    ]
    results = [(name, run_case(spec, _config())) for name, spec in modes]
    print(
        Report().render(
            [{"label": name, **result.metrics.as_row()} for name, result in results]
        )
    )
    print("\n  Burst offers everything at once, so its TTFT is nearly all queueing.\n"
          "  Closed-loop never queues at all, which is what makes it the right\n"
          "  workload for comparing scheduler behaviour rather than capacity.")


def main() -> None:
    workload = base_workload()
    print("MiniServe benchmark demo")
    print(
        f"workload: {workload.num_requests} requests, chat prompts, balanced outputs, "
        f"seed={workload.seed}\n"
    )
    sweep_rate(workload)
    sweep_kv(workload)
    sweep_burst(workload)


if __name__ == "__main__":
    main()
