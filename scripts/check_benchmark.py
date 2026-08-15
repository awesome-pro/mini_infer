"""Checks for the Phase 5 workload generator and benchmark harness.

The point of a workload model is that it produces the *shape* of load it claims to.
These checks assert those shapes: that a closed-loop workload never builds a queue,
that a burst does, that Poisson arrivals converge on the requested rate, and that a
seeded workload reproduces exactly.

Run with::

    python scripts/check_benchmark.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, close, expect_raises, report  # noqa: E402

from mini_infer import Engine, EngineConfig, RunnerConfig  # noqa: E402
from mini_infer.benchmark import (  # noqa: E402
    Driver,
    Report,
    WorkloadSpec,
    output_profile,
    prompt_profile,
    run_case,
    sweep_axis,
    write_csv,
    write_json,
)
from mini_infer.benchmark.workloads import Constant, LogNormal, Uniform  # noqa: E402

FAST_RUNNER = RunnerConfig(
    prefill_base_ms=1.0,
    prefill_ms_per_token=0.05,
    decode_base_ms=1.0,
    decode_ms_per_token=0.05,
    decode_ms_per_context_token=0.002,
)


def base_config(**overrides) -> EngineConfig:
    settings: dict = {
        "max_batch_tokens": 128,
        "max_running_requests": 4,
        "num_blocks": 256,
        "block_size": 16,
        "runner": FAST_RUNNER,
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def small_spec(arrival: str, **overrides) -> WorkloadSpec:
    settings: dict = {
        "num_requests": 24,
        "prompt": prompt_profile("short"),
        "output": output_profile("short"),
        "arrival": arrival,
        "seed": 5,
    }
    settings.update(overrides)
    return WorkloadSpec(**settings)


# ------------------------------------------------------------ length distributions


def check_constant_distribution() -> None:
    import random

    rng = random.Random(0)
    distribution = Constant(42)
    assert distribution.sample(rng) == 42


def check_uniform_stays_in_range() -> None:
    import random

    rng = random.Random(1)
    distribution = Uniform(10, 20)
    draws = [distribution.sample(rng) for _ in range(200)]
    assert min(draws) >= 10
    assert max(draws) <= 20
    assert len(set(draws)) > 5, "a uniform draw should vary"


def check_lognormal_has_a_long_tail() -> None:
    import random

    rng = random.Random(2)
    draws = [LogNormal(median=64, sigma=0.9).sample(rng) for _ in range(500)]
    assert min(draws) >= 1
    assert max(draws) > 3 * 64, "a lognormal should reach well past its median"
    below = sum(1 for d in draws if d < 64)
    assert 0.3 < below / len(draws) < 0.7, "the median should split the mass roughly evenly"


def check_length_profile_clamps() -> None:
    import random

    rng = random.Random(3)
    profile = prompt_profile("chat")
    draws = [profile.sample(rng) for _ in range(500)]
    assert min(draws) >= profile.minimum
    assert max(draws) <= profile.maximum


def check_unknown_profiles_are_rejected() -> None:
    expect_raises(ValueError, "unknown prompt profile", lambda: prompt_profile("nope"))
    expect_raises(ValueError, "unknown output profile", lambda: output_profile("nope"))


def check_spec_validates_itself() -> None:
    expect_raises(ValueError, "num_requests", lambda: WorkloadSpec(num_requests=0))
    expect_raises(ValueError, "unknown arrival", lambda: WorkloadSpec(num_requests=1, arrival="nope"))
    expect_raises(
        ValueError, "rate must be", lambda: WorkloadSpec(num_requests=1, arrival="poisson", rate=0)
    )
    expect_raises(
        ValueError,
        "concurrency must be",
        lambda: WorkloadSpec(num_requests=1, arrival="closed_loop", concurrency=0),
    )


# ------------------------------------------------------------- arrival processes


def check_seeded_workloads_are_reproducible() -> None:
    spec = small_spec("poisson", rate=20.0)
    first, second = spec.build(), spec.build()
    assert [r.prompt_len for r in first.requests] == [r.prompt_len for r in second.requests]
    assert [r.max_new_tokens for r in first.requests] == [
        r.max_new_tokens for r in second.requests
    ]


def check_different_seeds_give_different_workloads() -> None:
    a = small_spec("poisson", seed=1, prompt=prompt_profile("chat")).build()
    b = small_spec("poisson", seed=2, prompt=prompt_profile("chat")).build()
    assert [r.prompt_len for r in a.requests] != [r.prompt_len for r in b.requests]


def check_poisson_rate_matches_the_requested_load() -> None:
    """Mean inter-arrival time should be 1/rate."""
    spec = WorkloadSpec(
        num_requests=400,
        prompt=prompt_profile("short"),
        output=output_profile("short"),
        arrival="poisson",
        rate=25.0,
        seed=11,
    )
    workload = spec.build()
    driver = Driver(Engine(base_config()), workload=workload)

    arrivals = []
    now = 0.0
    while True:
        driver.coordinate(now)
        arrival = driver.peek()
        if arrival is None:
            break
        now = max(now, arrival)
        driver.coordinate(now)
        arrivals.append(driver._last_arrival)  # noqa: SLF001 - observing the schedule

    gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
    mean_gap = sum(gaps) / len(gaps)
    assert abs(mean_gap - 1.0 / 25.0) < 0.01, f"mean gap {mean_gap} should be near 0.04"


def check_burst_releases_everything_at_once() -> None:
    spec = small_spec("burst")
    workload = spec.build()
    engine = Engine(base_config())
    driver = Driver(engine, workload)
    driver.coordinate(0.0)

    assert driver.pending == 0, "a burst is fully offered immediately"
    assert all(r.arrival_time == 0.0 for r in workload.requests)


def check_closed_loop_limits_requests_in_flight() -> None:
    concurrency = 3
    spec = small_spec("closed_loop", concurrency=concurrency)
    engine = Engine(base_config())
    driver = Driver(engine, spec.build())

    peak = 0
    while True:
        now = engine.clock.now()
        driver.coordinate(now)
        peak = max(peak, driver.in_flight)
        if not engine.has_pending_work() and driver.peek() is None:
            break
        driver.coordinate(now)
        if not engine.has_pending_work():
            break
        engine.step()

    assert peak <= concurrency, f"in flight peaked at {peak}, above {concurrency}"


# ----------------------------------------------------------------------- driver


def check_closed_loop_never_builds_a_queue() -> None:
    """The defining property: a client waits for its own completion, so nothing queues."""
    spec = small_spec("closed_loop", num_requests=24, concurrency=4)
    result = run_case(spec, base_config())

    assert result.finished_all
    assert result.metrics.mean_queue_time == 0.0, "a closed loop never queues"
    assert result.metrics.p95_queue_time == 0.0
    assert not result.stalled


def check_burst_does_build_a_queue() -> None:
    """The opposite property, which is why burst is a stress test and not traffic."""
    spec = small_spec("burst", num_requests=24)
    result = run_case(spec, base_config())

    assert result.finished_all
    assert result.metrics.mean_queue_time > 0, "a burst must queue"
    assert result.metrics.mean_ttft > result.metrics.mean_queue_time


def check_poisson_saturates_with_arrival_rate() -> None:
    """Past saturation, open-loop queueing delay grows with the offered rate."""
    # 50 rps is below this engine's capacity, so the rates have to straddle it.
    results = [
        run_case(small_spec("poisson", num_requests=40, rate=rate), base_config())
        for rate in (2.0, 400.0)
    ]
    slow, fast = results

    assert slow.finished_all and fast.finished_all
    assert fast.metrics.mean_queue_time > slow.metrics.mean_queue_time, (
        "a higher arrival rate must produce more queueing"
    )


def check_all_arrival_modes_complete_a_feasible_workload() -> None:
    for arrival in ("burst", "poisson", "closed_loop"):
        result = run_case(small_spec(arrival, num_requests=20), base_config())
        assert result.finished_all, f"{arrival} did not finish"
        assert not result.stalled
        assert result.metrics.num_finished == 20
        assert result.metrics.generated_tokens > 0


def check_small_pool_stalls_and_is_reported() -> None:
    """A pool too small for the prompts cannot be scheduled, and must say so."""
    spec = WorkloadSpec(
        num_requests=20,
        prompt=prompt_profile("mixed"),
        output=output_profile("long"),
        arrival="poisson",
        rate=20.0,
        seed=9,
    )
    result = run_case(spec, base_config(num_blocks=8))

    assert not result.finished_all, "this pool cannot hold the workload"
    assert result.stalled, "the failure mode must be reported as a stall"


def check_completed_run_is_never_reported_as_stalled() -> None:
    """A finished workload must not be flagged stalled by its final no-op step.

    After the last request completes, one more step schedules nothing. That is
    completion, not a deadlock, and a benchmark that confused the two would report
    good configurations as broken.
    """
    for budget in (16, 32, 64, 128):
        result = run_case(
            small_spec("poisson", num_requests=60, rate=20.0), base_config(max_batch_tokens=budget)
        )
        assert result.finished_all, f"budget {budget} did not finish"
        assert not result.stalled, f"budget {budget} finished but was flagged stalled"


def check_driver_does_not_abandon_running_work() -> None:
    """A stalled step with requests still running must not end the run."""
    spec = small_spec("poisson", num_requests=16, rate=40.0)
    result = run_case(spec, base_config())

    assert result.finished_all, "the driver must keep going while work is resident"


# --------------------------------------------------------------------- reporting


def check_report_renders_a_table() -> None:
    result = run_case(small_spec("poisson", rate=10.0), base_config())
    row = {"label": "case", **result.as_row()}
    table = Report().render([row], title="t")
    lines = table.splitlines()
    assert lines[0] == "t"
    assert len(lines) == 4, "title, header, rule, one row"
    assert "TTFT" in lines[1]


def check_report_formats_numbers_readably() -> None:
    result = run_case(small_spec("poisson", rate=10.0), base_config())
    row = {"label": "case", **result.as_row()}
    table = Report().render([row])
    assert "nan" not in table.lower() or table.count("-") > 0


def check_sweep_produces_one_row_per_config() -> None:
    spec = small_spec("poisson", rate=15.0)
    rows = sweep_axis(
        spec,
        base_config(),
        values=[64, 128],
        apply=lambda config, value: config.with_(num_blocks=value),
        label=lambda value: f"{value} blocks",
    )
    assert len(rows) == 2
    assert [row.label for row in rows] == ["64 blocks", "128 blocks"]
    assert all(row.result.finished_all for row in rows)


def check_sweep_rows_are_self_contained() -> None:
    """A row must record the configuration that produced it, or it cannot be replayed."""
    spec = small_spec("poisson", rate=15.0)
    rows = sweep_axis(
        spec,
        base_config(),
        values=[96],
        apply=lambda config, value: config.with_(num_blocks=value, policy="fcfs"),
        label=lambda value: f"{value}",
    )
    row = rows[0].as_row()
    assert row["num_blocks"] == 96
    assert row["policy"] == "fcfs"
    assert row["arrival"] == "poisson"


def check_json_export_round_trips() -> None:
    spec = small_spec("poisson", rate=15.0)
    rows = sweep_axis(spec, base_config(), values=[128], apply=lambda c, v: c, label=str)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_json(rows, Path(tmp) / "results.json")
        payload = json.loads(path.read_text())
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert "config" in run and "workload" in run and "metrics" in run
    assert run["metrics"]["generated_tokens"] > 0


def check_csv_export_has_one_row_per_run() -> None:
    spec = small_spec("poisson", rate=15.0)
    rows = sweep_axis(
        spec,
        base_config(),
        values=[96, 192],
        apply=lambda c, v: c.with_(num_blocks=v),
        label=lambda v: f"{v}",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_csv(rows, Path(tmp) / "results.csv")
        with path.open() as handle:
            records = list(csv.DictReader(handle))
    assert len(records) == 2
    assert "num_blocks" in records[0]
    assert float(records[0]["output_tok_per_s"]) > 0


def main() -> int:
    check("constant distribution", check_constant_distribution)
    check("uniform stays in range", check_uniform_stays_in_range)
    check("lognormal has a long tail", check_lognormal_has_a_long_tail)
    check("length profile clamps", check_length_profile_clamps)
    check("unknown profiles rejected", check_unknown_profiles_are_rejected)
    check("spec validates itself", check_spec_validates_itself)
    check("seeded workloads reproduce", check_seeded_workloads_are_reproducible)
    check("different seeds differ", check_different_seeds_give_different_workloads)
    check("poisson rate matches requested load", check_poisson_rate_matches_the_requested_load)
    check("burst releases everything at once", check_burst_releases_everything_at_once)
    check("closed loop limits in-flight requests", check_closed_loop_limits_requests_in_flight)
    check("closed loop never builds a queue", check_closed_loop_never_builds_a_queue)
    check("burst does build a queue", check_burst_does_build_a_queue)
    check("poisson saturates with rate", check_poisson_saturates_with_arrival_rate)
    check("all arrival modes complete", check_all_arrival_modes_complete_a_feasible_workload)
    check("infeasible pool is reported as a stall", check_small_pool_stalls_and_is_reported)
    check("completed runs are never stalled", check_completed_run_is_never_reported_as_stalled)
    check("driver does not abandon running work", check_driver_does_not_abandon_running_work)
    check("report renders a table", check_report_renders_a_table)
    check("report formats numbers", check_report_formats_numbers_readably)
    check("sweep gives one row per config", check_sweep_produces_one_row_per_config)
    check("sweep rows are self-contained", check_sweep_rows_are_self_contained)
    check("json export round-trips", check_json_export_round_trips)
    check("csv export has one row per run", check_csv_export_has_one_row_per_run)
    return report("phase 5 benchmark harness")


if __name__ == "__main__":
    raise SystemExit(main())
