"""Benchmark CLI.

Runs a workload against the engine and prints a results table, optionally sweeping
one axis and exporting the rows for a README or a plot.

Examples::

    python -m mini_infer.cli --requests 200 --arrival poisson --rate 12
    python -m mini_infer.cli --sweep rate --values 2,4,8,16,32,64
    python -m mini_infer.cli --sweep policy --values fcfs,prefill_first,balanced,static
    python -m mini_infer.cli --sweep num_blocks --values 16,32,64,128 --export results
"""

from __future__ import annotations

import argparse
import sys

from dataclasses import replace

from mini_infer.benchmark.benchmark import (
    BenchmarkCase,
    Report,
    SweepRow,
    write_csv,
    write_json,
)
from mini_infer.benchmark.workloads import (
    OUTPUT_PROFILES,
    PROMPT_PROFILES,
    WorkloadSpec,
    output_profile,
    prompt_profile,
)
from mini_infer.config import EngineConfig
from mini_infer.engine.policies import policy_names

#: Configuration axes that can be swept, and how to apply a value.
SWEEPS: dict[str, str] = {
    "policy": "scheduling policy",
    "rate": "requests per second for poisson arrivals",
    "num_blocks": "KV pool size in blocks",
    "block_size": "tokens per KV block",
    "max_batch_tokens": "per-step token budget",
    "max_running_requests": "concurrent request limit",
    "max_prefill_chunk": "largest prefill chunk",
    "max_wait_steps": "balanced: ageing bound in steps",
    "prefill_reservation": "balanced: tokens per step reserved for prefill",
    "arrival": "burst, poisson or closed_loop",
}

#: Sweep axes whose values are names rather than numbers.
STRING_AXES = ("policy", "arrival")


def parse_values(raw: str, axis: str) -> list:
    if axis in STRING_AXES:
        return [part.strip() for part in raw.split(",") if part.strip()]
    values: list = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part) if part.lstrip("-").isdigit() else float(part))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-infer",
        description="Benchmark the mini inference runtime.",
    )
    parser.add_argument("--requests", type=int, default=200, help="number of requests")
    parser.add_argument(
        "--arrival",
        default="poisson",
        choices=("burst", "poisson", "closed_loop"),
        help="arrival process",
    )
    parser.add_argument("--rate", type=float, default=16.0, help="requests/sec for poisson")
    parser.add_argument("--concurrency", type=int, default=8, help="clients for closed_loop")
    parser.add_argument("--seed", type=int, default=0, help="workload seed")
    parser.add_argument(
        "--prompt",
        default="chat",
        choices=sorted(PROMPT_PROFILES),
        help="prompt length profile",
    )
    parser.add_argument(
        "--output",
        default="balanced",
        choices=sorted(OUTPUT_PROFILES),
        help="output length profile",
    )

    parser.add_argument(
        "--policy",
        default="fcfs",
        choices=policy_names(),
        help="scheduling policy",
    )
    parser.add_argument(
        "--max-wait-steps",
        type=int,
        default=16,
        help="balanced: steps before the oldest waiter is served first (0 disables)",
    )
    parser.add_argument(
        "--prefill-reservation",
        type=int,
        default=16,
        help="balanced: tokens per step decodes may not spend",
    )

    parser.add_argument("--max-batch-tokens", type=int, default=64)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-prefill-chunk", type=int, default=None)
    parser.add_argument("--no-chunked-prefill", action="store_true")
    parser.add_argument("--no-preemption", action="store_true")

    parser.add_argument("--sweep", choices=sorted(SWEEPS), help="axis to vary")
    parser.add_argument("--values", help="comma-separated values for --sweep")
    parser.add_argument("--export", help="path prefix for .json and .csv exports")
    parser.add_argument("--max-steps", type=int, default=None, help="hard step limit")
    parser.add_argument("--json", action="store_true", help="emit the table as JSON")
    return parser


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        policy=args.policy,
        max_batch_tokens=args.max_batch_tokens,
        max_running_requests=args.max_running_requests,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_prefill_chunk=args.max_prefill_chunk,
        enable_chunked_prefill=not args.no_chunked_prefill,
        enable_preemption=not args.no_preemption,
        max_wait_steps=args.max_wait_steps,
        prefill_reservation=args.prefill_reservation,
    )


def workload_from_args(args: argparse.Namespace, **overrides) -> WorkloadSpec:
    settings: dict = {
        "num_requests": args.requests,
        "prompt": prompt_profile(args.prompt),
        "output": output_profile(args.output),
        "arrival": args.arrival,
        "rate": args.rate,
        "concurrency": args.concurrency,
        "seed": args.seed,
    }
    settings.update(overrides)
    return WorkloadSpec(**settings)


def apply_sweep(
    config: EngineConfig, workload: WorkloadSpec, axis: str, value
) -> tuple[EngineConfig, WorkloadSpec]:
    """The (config, workload) pair for one point of a sweep."""
    if axis == "policy":
        return config.with_(policy=str(value)), workload
    if axis == "rate":
        return config, replace(workload, rate=float(value))
    if axis == "arrival":
        return config, replace(workload, arrival=str(value))
    if axis == "max_prefill_chunk":
        return config.with_(max_prefill_chunk=int(value)), workload
    return config.with_(**{axis: int(value)}), workload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_workload = workload_from_args(args)
    base_config = config_from_args(args)

    if args.sweep and not args.values:
        print("--sweep requires --values", file=sys.stderr)
        return 2

    if args.sweep:
        points = [
            (*apply_sweep(base_config, base_workload, args.sweep, value), str(value))
            for value in parse_values(args.values, args.sweep)
        ]
    else:
        points = [(base_config, base_workload, "default")]

    runs: list[SweepRow] = []
    for config, workload, label in points:
        result = BenchmarkCase(
            label=label, workload=workload, config=config
        ).run(max_steps=args.max_steps)
        runs.append(SweepRow(label=label, config=config, result=result))

    rows = []
    for run in runs:
        row = run.as_row()
        row["completed"] = f"{row['finished']}/{row['requests']}"
        rows.append(row)

    header = (
        f"{args.requests} requests, {args.arrival} arrivals, "
        f"prompt={args.prompt}, output={args.output}, seed={args.seed}"
    )
    print(Report().render(rows, title=header))
    print(
        f"\nengine: policy={base_config.policy} "
        f"max_batch_tokens={base_config.max_batch_tokens} "
        f"max_running_requests={base_config.max_running_requests} "
        f"num_blocks={base_config.num_blocks} block_size={base_config.block_size} "
        f"chunked_prefill={base_config.enable_chunked_prefill} "
        f"preemption={base_config.enable_preemption}"
    )
    if base_config.policy == "balanced":
        print(
            f"balanced: prefill_reservation={base_config.prefill_reservation} "
            f"max_wait_steps={base_config.max_wait_steps}"
        )
    stalled = [row for row in rows if row.get("stalled")]
    if stalled:
        names = ", ".join(str(row.get("label")) for row in stalled)
        print(
            f"\n{len(stalled)} run(s) stalled ({names}): the KV pool cannot hold this\n"
            "workload, so requests hold partial sequences and cannot advance. Raise\n"
            "--num-blocks, shorten the prompts, or lower --max-running-requests."
        )

    if args.json:
        import json

        print(json.dumps(rows, indent=2, default=str))

    if args.export:
        print(f"\nwrote {write_json(runs, args.export + '.json')}")
        print(f"wrote {write_csv(runs, args.export + '.csv')}")
    return 1 if stalled else 0


if __name__ == "__main__":
    raise SystemExit(main())
