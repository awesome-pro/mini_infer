"""Generating the README's figures from real runs.

Every figure in ``figures/`` comes from this script, so a number in the README can
always be traced back to a command. Nothing here is hand-drawn or illustrative: each
figure is a benchmark that was actually executed, with a fixed seed and the modelled
cost model.

Run with::

    python scripts/make_figures.py                 # writes figures/
    python scripts/make_figures.py --out /tmp/figs  # somewhere else
    python scripts/make_figures.py --only policies  # one experiment
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, EngineConfig, Request, policy_names
from mini_infer.benchmark.benchmark import BenchmarkCase
from mini_infer.benchmark.driver import Driver
from mini_infer.benchmark.workloads import (
    Constant,
    LengthProfile,
    LogNormal,
    Uniform,
    WorkloadSpec,
)
from mini_infer.visualizations.plots import (
    HistogramPanel,
    Panel,
    Point,
    plot_bars,
    plot_histograms,
    plot_lines,
    plot_scatter,
    plot_step_series,
    plot_timeline,
    series_from_rows,
    step_series,
    timeline_grid,
)
from mini_infer.visualizations.timeline import timeline_lines

#: Guards a sweep against a configuration that cannot drain its pool.
MAX_STEPS = 40_000


def _profile(value) -> LengthProfile:
    """Accept a fixed length, a length distribution, or an already-built profile."""
    if isinstance(value, LengthProfile):
        return value
    if isinstance(value, int):
        return LengthProfile(Constant(value))
    return LengthProfile(value)


def spec(num_requests: int, prompt, output, **kwargs) -> WorkloadSpec:
    return WorkloadSpec(
        num_requests=num_requests,
        prompt=_profile(prompt),
        output=_profile(output),
        **kwargs,
    )


MIXED_PROMPT = LengthProfile(LogNormal(median=64, sigma=0.9), minimum=4, maximum=512)
BALANCED_OUTPUT = LengthProfile(LogNormal(median=48, sigma=0.7), minimum=4, maximum=256)
#: Uniform 32-512 token prompts: long enough that the step budget actually binds.
LONG_MIX_PROMPT = LengthProfile(Uniform(32, 512))


def mixed_spec(num_requests: int, **arrivals) -> WorkloadSpec:
    """The mixed-length workload most experiments share, with arrivals varied."""
    return spec(num_requests, MIXED_PROMPT, BALANCED_OUTPUT, **arrivals)


def knee(values: Sequence[float], *, tolerance: float = 0.99) -> int:
    """First index that reaches within ``tolerance`` of the curve's maximum.

    Marking the maximum instead would point at the last point of a flattening curve
    and read as though that were where the knee is.
    """
    peak = max(values)
    for index, value in enumerate(values):
        if value >= peak * tolerance:
            return index
    return len(values) - 1


def run(label: str, workload: WorkloadSpec, config: EngineConfig) -> dict:
    """One benchmark point, as a flat row plus its label."""
    result = BenchmarkCase(label=label, workload=workload, config=config).run(
        max_steps=MAX_STEPS
    )
    return {"label": label, **result.as_row()}


def echo(rows: list[dict], columns: list[str], title: str) -> None:
    """Print the columns the README will quote."""
    headers = [column.replace("_ms", "").replace("_tok_per_s", "") for column in columns]
    widths = [
        max(len(headers[i]), *(len(_cell(row.get(column))) for row in rows))
        for i, column in enumerate(columns)
    ]
    print(f"\n{title}")
    print("  ".join(header.rjust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        cells = (_cell(row.get(column)).rjust(widths[i]) for i, column in enumerate(columns))
        print("  ".join(cells))


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) >= 100 else f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# ------------------------------------------------------------------ figures


def figure_timeline(out: Path) -> None:
    """The scheduler timeline, and the same run's KV and load traces."""
    print("\n=== timeline: 4 requests, budget 32, chunk cap 16, 24 blocks ===")
    workload = spec(4, Uniform(16, 56), 10, arrival="burst", seed=3)
    config = EngineConfig(
        max_batch_tokens=32,
        max_running_requests=4,
        num_blocks=24,
        block_size=16,
        max_prefill_chunk=16,
    )
    engine = Engine(config)
    result = Driver(engine, workload.build()).run()
    requests = list(engine.requests.values())
    grid = timeline_grid(engine.history, requests)

    print("\n".join(timeline_lines(engine.history, requests, max_rows=40)))
    print(f"steps={result.steps} finished={result.metrics.num_finished}/4")

    plot_timeline(grid, out / "timeline.png", title="scheduler timeline: one row per request")
    plot_step_series(
        step_series(engine), out / "kv_and_load.png", title="KV pressure and work mix over one run"
    )


def figure_latency_distributions(out: Path) -> None:
    print("\n=== latency distributions: 300 requests, poisson 16/s ===")
    workload = mixed_spec(300, arrival="poisson", rate=16.0, seed=0)
    config = EngineConfig(max_batch_tokens=64, max_running_requests=4, num_blocks=256)
    engine = Engine(config)
    result = Driver(engine, workload.build()).run()

    tracked = [m for m in engine.metrics.requests.values() if m.finished]
    ttfts = [m.ttft * 1e3 for m in tracked if m.ttft == m.ttft]
    itls = [gap * 1e3 for m in tracked for gap in m.inter_token_latencies]
    metrics = result.metrics
    print(
        f"ttft mean={metrics.mean_ttft * 1e3:.1f}ms p50={metrics.p50_ttft * 1e3:.1f} "
        f"p95={metrics.p95_ttft * 1e3:.1f} | itl mean={metrics.mean_itl * 1e3:.2f} "
        f"p95={metrics.p95_itl * 1e3:.2f} | out={metrics.output_throughput:.0f} tok/s"
    )

    plot_histograms(
        [
            HistogramPanel(
                title="time to first token",
                xlabel="TTFT (ms)",
                values=ttfts,
                ylabel="requests",
                reference=metrics.p95_ttft * 1e3,
                reference_text=f"p95 {metrics.p95_ttft * 1e3:,.0f} ms",
            ),
            HistogramPanel(
                title="inter-token latency",
                xlabel="ITL (ms)",
                values=itls,
                ylabel="intervals",
                reference=metrics.p95_itl * 1e3,
                reference_text=f"p95 {metrics.p95_itl * 1e3:,.1f} ms",
            ),
            HistogramPanel(
                title="end-to-end latency",
                xlabel="E2E (ms)",
                values=[m.e2e * 1e3 for m in tracked],
                ylabel="requests",
                reference=metrics.p95_e2e * 1e3,
                reference_text=f"p95 {metrics.p95_e2e * 1e3:,.0f} ms",
            ),
        ],
        out / "latency_distributions.png",
        title="one run's latency spread, not just its mean",
    )


def figure_saturation(out: Path) -> None:
    print("\n=== saturation: throughput and latency against offered rate ===")
    rates = [2, 4, 8, 16, 32, 64, 128, 256]
    rows = [
        run(
            f"{rate}/s",
            mixed_spec(120, arrival="poisson", rate=float(rate), seed=0),
            EngineConfig(max_batch_tokens=64, max_running_requests=4, num_blocks=256),
        )
        for rate in rates
    ]
    echo(
        rows,
        ["label", "finished", "output_tok_per_s", "mean_ttft_ms", "p95_ttft_ms", "p95_queue_ms"],
        "rate sweep",
    )

    throughput = series_from_rows(rows, "output_tok_per_s")
    saturated_at = knee(throughput)
    plot_lines(
        rates,
        [
            Panel(
                "throughput",
                "tokens/s",
                {"output": throughput},
                mark=saturated_at,
                mark_text=f"saturates near {rates[saturated_at]}/s",
            ),
            Panel("time to first token", "ms", {"p95": series_from_rows(rows, "p95_ttft_ms")}),
            Panel("queueing delay", "ms", {"p95": series_from_rows(rows, "p95_queue_ms")}),
        ],
        out / "saturation.png",
        xlabel="offered load (requests/s)",
        title="capacity: throughput flattens while queueing grows",
        log_x=True,
    )


def figure_arrival_models(out: Path) -> None:
    print("\n=== arrival models at the same offered load ===")
    cases = (
        ("burst", mixed_spec(120, arrival="burst", seed=0)),
        ("poisson 16/s", mixed_spec(120, arrival="poisson", rate=16.0, seed=0)),
        ("closed loop (4)", mixed_spec(120, arrival="closed_loop", concurrency=4, seed=0)),
    )
    rows = [
        run(
            label,
            workload,
            EngineConfig(max_batch_tokens=64, max_running_requests=4, num_blocks=256),
        )
        for label, workload in cases
    ]
    echo(
        rows,
        ["label", "finished", "output_tok_per_s", "p95_ttft_ms", "mean_queue_ms", "p95_itl_ms"],
        "arrival models",
    )

    groups = [row["label"] for row in rows]
    plot_bars(
        groups,
        [
            Panel("time to first token", "ms", {"p95": series_from_rows(rows, "p95_ttft_ms")}),
            Panel("queueing delay", "ms", {"mean": series_from_rows(rows, "mean_queue_ms")}),
            Panel("throughput", "tokens/s", {"output": series_from_rows(rows, "output_tok_per_s")}),
        ],
        out / "arrival_models.png",
        title="the arrival process changes what the same load measures",
    )


def figure_budget(out: Path) -> None:
    print("\n=== token budget: bigger is not better ===")
    budgets = [16, 32, 64, 128]
    rows = [
        run(
            str(budget),
            spec(150, LONG_MIX_PROMPT, BALANCED_OUTPUT, arrival="poisson", rate=12.0, seed=0),
            EngineConfig(max_batch_tokens=budget, max_running_requests=4, num_blocks=256),
        )
        for budget in budgets
    ]
    echo(
        rows,
        ["label", "output_tok_per_s", "mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_itl_ms"],
        "budget sweep",
    )

    ttft = series_from_rows(rows, "mean_ttft_ms")
    best = min(range(len(ttft)), key=ttft.__getitem__)
    plot_lines(
        budgets,
        [
            Panel("time to first token", "ms", {"mean": ttft}, mark=best, mark_text="best"),
            Panel("throughput", "tokens/s", {"output": series_from_rows(rows, "output_tok_per_s")}),
            Panel("inter-token latency", "ms", {"p95": series_from_rows(rows, "p95_itl_ms")}),
        ],
        out / "budget.png",
        xlabel="max_batch_tokens per step",
        title="the step budget trades latency against how much a step can hold",
        log_x=True,
    )


def figure_block_size(out: Path) -> None:
    """Experiment E, done twice: holding capacity constant, then holding the pool."""
    print("\n=== block size: fixed capacity versus fixed pool ===")
    sizes = [8, 16, 32, 64]
    capacity = 2048
    rows = []
    for size in sizes:
        workload = mixed_spec(120, arrival="poisson", rate=16.0, seed=0)
        rows.append(
            run(
                f"capacity {size}",
                workload,
                EngineConfig(
                    max_batch_tokens=64,
                    max_running_requests=4,
                    num_blocks=capacity // size,
                    block_size=size,
                ),
            )
        )
    for size in sizes:
        workload = mixed_spec(120, arrival="poisson", rate=16.0, seed=0)
        rows.append(
            run(
                f"pool {size}",
                workload,
                EngineConfig(
                    max_batch_tokens=64, max_running_requests=4, num_blocks=128, block_size=size
                ),
            )
        )
    echo(
        rows,
        [
            "label",
            "finished",
            "mean_fragmentation",
            "max_fragmentation",
            "p95_ttft_ms",
            "output_tok_per_s",
        ],
        "block size sweep",
    )

    constant = [row for row in rows if row["label"].startswith("capacity")]
    fixed = [row for row in rows if row["label"].startswith("pool")]
    plot_lines(
        sizes,
        [
            Panel(
                "internal fragmentation",
                "share of reserved KV",
                {
                    "capacity held constant": series_from_rows(constant, "mean_fragmentation"),
                    "pool size held constant": series_from_rows(fixed, "mean_fragmentation"),
                },
            ),
            Panel(
                "time to first token",
                "ms",
                {
                    "capacity held constant": series_from_rows(constant, "p95_ttft_ms"),
                    "pool size held constant": series_from_rows(fixed, "p95_ttft_ms"),
                },
            ),
            Panel(
                "throughput",
                "tokens/s",
                {
                    "capacity held constant": series_from_rows(constant, "output_tok_per_s"),
                    "pool size held constant": series_from_rows(fixed, "output_tok_per_s"),
                },
            ),
        ],
        out / "block_size.png",
        xlabel="tokens per block",
        title="block size: fragmentation is real, capacity matters more",
        log_x=True,
    )


def figure_kv_pressure(out: Path) -> None:
    """Experiment D: what a small KV pool costs."""
    print("\n=== KV pressure: pool size against queueing, evictions and completion ===")
    pools = [8, 16, 32, 64, 128, 256]
    rows = [
        run(
            str(blocks),
            spec(120, 48, 64, arrival="poisson", rate=16.0, seed=0),
            EngineConfig(max_batch_tokens=64, max_running_requests=16, num_blocks=blocks),
        )
        for blocks in pools
    ]
    echo(
        rows,
        [
            "label",
            "finished",
            "stalled",
            "output_tok_per_s",
            "p95_ttft_ms",
            "peak_kv_utilization",
            "preemptions",
        ],
        "pool sweep",
    )

    plot_lines(
        pools,
        [
            Panel("throughput", "tokens/s", {"output": series_from_rows(rows, "output_tok_per_s")}),
            Panel("time to first token", "ms", {"p95": series_from_rows(rows, "p95_ttft_ms")}),
            Panel("preemptions", "evictions", {"total": series_from_rows(rows, "preemptions")}),
            Panel("completed", "requests", {"finished": series_from_rows(rows, "finished")}),
        ],
        out / "kv_pressure.png",
        xlabel="KV pool (blocks of 16 tokens)",
        title="KV pressure: the pool sets how much can be in flight at once",
        log_x=True,
    )


def figure_chunking(out: Path) -> None:
    """Experiment B: chunked versus unchunked prefill."""
    print("\n=== chunked prefill: chunk size against TTFT and ITL ===")
    cases: list[tuple[str, EngineConfig]] = [
        (
            "budget (no cap)",
            EngineConfig(
                max_batch_tokens=64, max_running_requests=4, num_blocks=256,
                enable_chunked_prefill=False,
            ),
        )
    ]
    cases += [
        (
            f"cap {cap}",
            EngineConfig(
                max_batch_tokens=64, max_running_requests=4, num_blocks=256, max_prefill_chunk=cap
            ),
        )
        for cap in (16, 32, 64)
    ]
    rows = [
        run(
            label,
            mixed_spec(150, arrival="poisson", rate=16.0, seed=0),
            config,
        )
        for label, config in cases
    ]
    echo(
        rows,
        [
            "label",
            "output_tok_per_s",
            "mean_ttft_ms",
            "p95_ttft_ms",
            "p95_itl_ms",
            "max_prefill_tokens",
        ],
        "chunk sweep",
    )

    groups = [row["label"] for row in rows]
    plot_bars(
        groups,
        [
            Panel("time to first token", "ms", {"mean": series_from_rows(rows, "mean_ttft_ms")}),
            Panel("inter-token latency", "ms", {"p95": series_from_rows(rows, "p95_itl_ms")}),
            Panel("throughput", "tokens/s", {"output": series_from_rows(rows, "output_tok_per_s")}),
        ],
        out / "chunking.png",
        title="chunked prefill: smoothing decode at the cost of admission latency",
    )


def figure_static_vs_continuous(out: Path) -> None:
    """Experiment A: the headline continuous-batching result."""
    print("\n=== static versus continuous batching across offered load ===")
    rates = [2, 4, 8, 16, 32]
    rows = []
    for policy in ("fcfs", "static"):
        for rate in rates:
            rows.append(
                run(
                    f"{policy} {rate}/s",
                    mixed_spec(120, arrival="poisson", rate=float(rate), seed=0),
                    EngineConfig(
                        policy=policy,
                        max_batch_tokens=64,
                        max_running_requests=4,
                        num_blocks=256,
                    ),
                )
            )
    echo(
        rows,
        ["label", "finished", "output_tok_per_s", "mean_ttft_ms", "p95_ttft_ms", "p95_e2e_ms"],
        "static vs continuous",
    )

    continuous = rows[: len(rates)]
    static = rows[len(rates) :]
    plot_lines(
        rates,
        [
            Panel(
                "throughput",
                "tokens/s",
                {
                    "continuous": series_from_rows(continuous, "output_tok_per_s"),
                    "static": series_from_rows(static, "output_tok_per_s"),
                },
            ),
            Panel(
                "time to first token",
                "ms",
                {
                    "continuous": series_from_rows(continuous, "p95_ttft_ms"),
                    "static": series_from_rows(static, "p95_ttft_ms"),
                },
            ),
            Panel(
                "end-to-end latency",
                "ms",
                {
                    "continuous": series_from_rows(continuous, "p95_e2e_ms"),
                    "static": series_from_rows(static, "p95_e2e_ms"),
                },
            ),
        ],
        out / "static_vs_continuous.png",
        xlabel="offered load (requests/s)",
        title="continuous batching refills a freed slot; static batching waits for the batch",
        log_x=True,
    )


def figure_policies(out: Path) -> None:
    """The policy set on one saturating workload, plus the isolated trade-off."""
    print("\n=== policies on one saturating workload ===")
    workload = mixed_spec(200, arrival="poisson", rate=64.0, seed=0)
    compared = ("fcfs", "prefill_first", "balanced", "static")
    rows = [
        run(
            policy,
            workload,
            EngineConfig(
                policy=policy,
                max_batch_tokens=64,
                max_running_requests=8,
                num_blocks=256,
                prefill_reservation=16,
                max_wait_steps=16,
            ),
        )
        for policy in compared
    ]
    echo(
        rows,
        ["label", "finished", "output_tok_per_s", "mean_ttft_ms", "p95_ttft_ms", "p95_itl_ms"],
        "policies",
    )

    groups = [row["label"] for row in rows]
    plot_bars(
        groups,
        [
            Panel("time to first token", "ms", {"mean": series_from_rows(rows, "mean_ttft_ms")}),
            Panel("inter-token latency", "ms", {"p95": series_from_rows(rows, "p95_itl_ms")}),
            Panel("throughput", "tokens/s", {"output": series_from_rows(rows, "output_tok_per_s")}),
        ],
        out / "policies.png",
        title="one workload, four scheduling policies",
    )

    print("\n=== the same trade-off, isolated: one long prompt behind 8 decoders ===")
    stalls = {}
    late_ttfts = {}
    for policy in ("decode_first", "prefill_first", "balanced", "static"):
        late_ttft, stall = _late_prompt(policy)
        late_ttfts[policy] = late_ttft * 1e3
        stalls[policy] = stall * 1e3
        served = late_ttfts[policy] == late_ttfts[policy]
        shown = f"{late_ttfts[policy]:8.1f}ms" if served else " not served"
        print(f"  {policy:14} late_ttft={shown}  decoders_stalled={stalls[policy]:7.1f}ms")

    served = [label for label in late_ttfts if late_ttfts[label] == late_ttfts[label]]
    unserved = [label for label in late_ttfts if label not in served]
    if unserved:
        print(f"  (not served within 400 steps: {', '.join(unserved)})")
    plot_scatter(
        [Point(label, late_ttfts[label], stalls[label]) for label in served],
        out / "policy_tradeoff.png",
        xlabel="time to first token for the late prompt (ms, lower is better)",
        ylabel="delay to the existing decoders (ms, lower is better)",
        title="decode priority is a trade, not a win",
    )

    print("\n=== fairness: worst wait for a first token (40 requests, budget 8) ===")
    fairness_cases = (
        ("decode_first", "decode_first", {}),
        ("balanced floor only", "balanced", {"prefill_reservation": 2, "max_wait_steps": 0}),
        ("balanced ageing only", "balanced", {"prefill_reservation": 0, "max_wait_steps": 8}),
        ("balanced both", "balanced", {"prefill_reservation": 2, "max_wait_steps": 8}),
    )
    waits = []
    for label, policy, knobs in fairness_cases:
        waits.append({"label": label, "worst_wait": _worst_wait(policy, **knobs)})
    echo(waits, ["label", "worst_wait"], "fairness")

    plot_bars(
        [row["label"] for row in waits],
        [
            Panel(
                "worst wait for a first token",
                "engine steps",
                {"steps": [row["worst_wait"] for row in waits]},
            )
        ],
        out / "fairness.png",
        title="what each fairness knob bounds",
    )


def _late_prompt(policy: str) -> tuple[float, float]:
    """Run 8 decoders, then submit one long prompt behind them.

    Returns the late request's time to first token and how long the existing decoders
    waited for their next token after it arrived, both in seconds of modelled time.
    """
    engine = Engine(
        EngineConfig(
            policy=policy,
            max_batch_tokens=12,
            max_running_requests=16,
            num_blocks=256,
            block_size=16,
            prefill_reservation=4,
            max_wait_steps=2,
        )
    )
    for i in range(8):
        engine.submit(
            Request(prompt_tokens=list(range(4)), max_new_tokens=300, arrival_time=0.0, id=f"w{i}")
        )
    for _ in range(20):
        engine.step()

    submitted_at = engine.clock.now()
    late = engine.submit(
        Request(
            prompt_tokens=list(range(64)),
            max_new_tokens=4,
            arrival_time=submitted_at,
            id="late",
        )
    )
    warm_next: dict[str, float] = {}
    late_ttft = float("nan")
    for _ in range(400):
        step = engine.step()
        for work in step.output.work:
            if work.kind.value == "decode" and work.num_new_tokens and work.request_id != "late":
                warm_next.setdefault(work.request_id, engine.clock.now())
        if late.first_token_time is not None:
            late_ttft = late.first_token_time - submitted_at
            break
    return late_ttft, min(warm_next.values()) - submitted_at


def _worst_wait(policy: str, **knobs) -> int:
    """Steps the unluckiest request waited for any compute at all."""
    engine = Engine(
        EngineConfig(
            policy=policy,
            max_batch_tokens=8,
            max_running_requests=64,
            num_blocks=1024,
            block_size=16,
            **knobs,
        )
    )
    workload = spec(40, 8, 40, arrival="burst")
    first_seen: dict[str, int] = {}
    first_served: dict[str, int] = {}
    for step in engine.run(workload.build().requests, max_steps=4000):
        for request in engine.waiting:
            first_seen.setdefault(request.id, step.index)
        for work in step.output.work:
            if work.num_new_tokens > 0:
                first_served.setdefault(work.request_id, step.index)
    waits = [first_served[k] - v for k, v in first_seen.items() if k in first_served]
    return max(waits) if waits else 0


def figure_prefix_cache(out: Path) -> None:
    """Prefix caching: the same shared prefix, with and without reuse."""
    print("\n=== prefix caching: a shared prefix across arrival rates ===")
    rates = [2, 4, 8, 16, 32]
    prefix = 480
    off_rows, on_rows = [], []
    for rate in rates:
        for enabled, sink in ((False, off_rows), (True, on_rows)):
            workload = shared_prefix_workload(120, prefix=prefix, rate=float(rate))
            config = EngineConfig(
                max_batch_tokens=64,
                max_running_requests=4,
                num_blocks=256,
                enable_prefix_cache=enabled,
            )
            engine = Engine(config)
            result = Driver(engine, workload).run()
            metrics = result.metrics
            sink.append(
                {
                    "label": f"{'on' if enabled else 'off'} {rate}/s",
                    "rate": rate,
                    "prefill_tokens": sum(step.num_prefill_tokens for step in engine.history),
                    "reused": metrics.cached_prefix_tokens,
                    "hits": metrics.prefix_cache_hits,
                    "output_tok_per_s": metrics.output_throughput,
                    "mean_ttft_ms": metrics.mean_ttft * 1e3,
                    "p95_ttft_ms": metrics.p95_ttft * 1e3,
                    "mean_e2e_ms": metrics.mean_e2e * 1e3,
                }
            )
    echo(
        off_rows + on_rows,
        [
            "label",
            "prefill_tokens",
            "reused",
            "hits",
            "output_tok_per_s",
            "mean_ttft_ms",
            "mean_e2e_ms",
        ],
        "prefix cache sweep",
    )

    plot_lines(
        rates,
        [
            Panel(
                "prefill tokens computed",
                "tokens",
                {
                    "without cache": series_from_rows(off_rows, "prefill_tokens"),
                    "with cache": series_from_rows(on_rows, "prefill_tokens"),
                },
            ),
            Panel(
                "time to first token",
                "ms",
                {
                    "without cache": series_from_rows(off_rows, "mean_ttft_ms"),
                    "with cache": series_from_rows(on_rows, "mean_ttft_ms"),
                },
            ),
            Panel(
                "end-to-end latency",
                "ms",
                {
                    "without cache": series_from_rows(off_rows, "mean_e2e_ms"),
                    "with cache": series_from_rows(on_rows, "mean_e2e_ms"),
                },
            ),
        ],
        out / "prefix_cache.png",
        xlabel="offered load (requests/s)",
        title=f"prefix caching: a {prefix}-token prefix shared by every request",
        log_x=True,
    )


def shared_prefix_workload(num_requests: int, *, prefix: int, rate: float) -> WorkloadSpec:
    """Every request starts with the same prefix and ends with its own tail."""
    workload = spec(
        num_requests,
        Constant(prefix + 8),
        Constant(32),
        arrival="poisson",
        rate=rate,
        seed=0,
    ).build()
    shared = list(range(1, prefix + 1))
    for index, request in enumerate(workload.requests):
        tail = [10_000 + index * 8 + offset for offset in range(8)]
        request.prompt_tokens = shared + tail
    return workload


EXPERIMENTS = {
    "timeline": figure_timeline,
    "latency": figure_latency_distributions,
    "saturation": figure_saturation,
    "arrivals": figure_arrival_models,
    "budget": figure_budget,
    "block_size": figure_block_size,
    "kv_pressure": figure_kv_pressure,
    "chunking": figure_chunking,
    "static": figure_static_vs_continuous,
    "policies": figure_policies,
    "prefix_cache": figure_prefix_cache,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make-figures", description=__doc__)
    parser.add_argument("--out", default="figures", help="directory for the PNGs")
    parser.add_argument(
        "--only", action="append", choices=sorted(EXPERIMENTS), help="run one experiment"
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    selected = args.only or list(EXPERIMENTS)
    print(f"writing {len(selected)} figures to {out}/")
    for name in selected:
        EXPERIMENTS[name](out)
    written = sorted(path.name for path in out.glob("*.png"))
    print(f"\n{len(written)} figures: {', '.join(written)}")
    print(f"policies available for comparison: {', '.join(policy_names())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
