"""Checks for the Phase 4 metrics collector.

These pin the *definitions*, because a latency number that quietly means something
else is worse than no number: TTFT/TPOT/ITL follow the definitions vLLM publishes,
and every aggregate must be traceable back to a step or a token.

Run with::

    python scripts/check_metrics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, close, report  # noqa: E402

from mini_infer import Engine, EngineConfig, Request, RunnerConfig  # noqa: E402
from mini_infer.metrics import MetricsCollector, RunMetrics, percentile  # noqa: E402

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


def build(**overrides) -> Engine:
    settings: dict = {
        "max_batch_tokens": 64,
        "max_running_requests": 4,
        "num_blocks": 64,
        "block_size": 16,
        "runner": ZERO_COST,
    }
    settings.update(overrides)
    return Engine(EngineConfig(**settings))


# ---------------------------------------------------------------- percentiles


def check_percentile_matches_numpy_definition() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    close(percentile(values, 0.0), 1.0)
    close(percentile(values, 0.5), 2.5)
    close(percentile(values, 1.0), 4.0)
    close(percentile(values, 0.25), 1.75)
    close(percentile([5.0], 0.95), 5.0)
    assert math.isnan(percentile([], 0.5))


def check_percentile_rejects_bad_quantiles() -> None:
    try:
        percentile([1.0], 1.5)
    except ValueError as exc:
        assert "quantile" in str(exc)
        return
    raise AssertionError("expected a ValueError for quantile > 1")


# ------------------------------------------------------------------- per request


def check_ttft_is_arrival_to_first_token() -> None:
    engine = build()
    request = engine.submit(make_request(8, 4, req_id="r0"))

    engine.run_to_completion()
    metrics = engine.metrics.requests["r0"].ttft

    assert not math.isnan(metrics)
    close(metrics, request.first_token_time - request.arrival_time)
    assert metrics > 0, "TTFT must include the prefill, so it is strictly positive"


def check_ttft_includes_queueing_delay() -> None:
    """TTFT measures from arrival, so time spent waiting is part of it."""
    engine = build(max_running_requests=1)
    first = engine.submit(make_request(8, 10, req_id="first"))
    second = engine.submit(make_request(8, 2, req_id="second"))

    engine.run_to_completion()

    assert second.arrival_time == first.arrival_time, "both arrive together"
    assert engine.metrics.requests["second"].ttft > engine.metrics.requests["first"].ttft, (
        "the request admitted later must report a larger TTFT"
    )


def check_tpot_matches_the_published_definition() -> None:
    engine = build()
    engine.submit(make_request(8, 5, req_id="r0"))

    engine.run_to_completion()
    metrics = engine.metrics.requests["r0"]

    assert metrics.output_tokens == 5
    expected = (metrics.e2e - metrics.ttft) / (metrics.output_tokens - 1)
    close(metrics.tpot, expected)
    assert not math.isnan(metrics.tpot)


def check_tpot_is_undefined_for_a_single_token() -> None:
    """With one output token there is no inter-token interval to average."""
    engine = build()
    engine.submit(make_request(8, 1, req_id="r0"))

    engine.run_to_completion()

    assert math.isnan(engine.metrics.requests["r0"].tpot)


def check_itl_is_the_gap_between_consecutive_tokens() -> None:
    engine = build()
    engine.submit(make_request(8, 4, req_id="r0"))

    engine.run_to_completion()
    metrics = engine.metrics.requests["r0"]

    gaps = metrics.inter_token_latencies
    assert len(gaps) == metrics.output_tokens - 1
    for earlier, later, gap in zip(metrics.token_times, metrics.token_times[1:], gaps, strict=False):
        close(gap, later - earlier)


def check_queue_time_is_arrival_to_first_run() -> None:
    engine = build(max_running_requests=1)
    engine.submit(make_request(8, 12, req_id="holder"))
    waiting = engine.submit(make_request(8, 2, req_id="waiter"))

    engine.run_to_completion()
    metrics = engine.metrics.requests["waiter"]

    assert not math.isnan(metrics.queue_time)
    assert metrics.queue_time > 0, "it had to wait for the running slot"
    assert metrics.queue_time < metrics.ttft, (
        "queue time ends when the first step runs it, before its first token"
    )
    assert waiting.admitted_step is not None


def check_e2e_covers_arrival_to_completion() -> None:
    engine = build()
    request = engine.submit(make_request(8, 3, req_id="r0"))

    engine.run_to_completion()
    metrics = engine.metrics.requests["r0"]

    close(metrics.e2e, request.finish_time - request.arrival_time)
    assert metrics.e2e > metrics.ttft


# ------------------------------------------------------------------- run-level


def check_throughput_is_tokens_over_elapsed_time() -> None:
    engine = build()
    for i in range(4):
        engine.submit(make_request(8, 6, req_id=f"r{i}"))

    engine.run_to_completion()
    elapsed = engine.clock.now()
    summary = engine.metrics.summary(elapsed=elapsed)

    close(summary.output_throughput, summary.generated_tokens / elapsed)
    close(summary.total_throughput, (summary.prompt_tokens + summary.generated_tokens) / elapsed)
    assert summary.generated_tokens == 24
    assert summary.prompt_tokens == 32


def check_generated_token_count_matches_the_engine() -> None:
    engine = build()
    for i in range(3):
        engine.submit(make_request(4 + i, 5, req_id=f"r{i}"))

    engine.run_to_completion()
    summary = engine.metrics.summary(elapsed=engine.clock.now())

    assert summary.generated_tokens == sum(r.num_generated for r in engine.finished)
    assert summary.num_finished == 3
    assert summary.prompt_tokens == sum(r.prompt_len for r in engine.finished)


def check_kv_utilization_comes_from_step_state() -> None:
    engine = build(num_blocks=8, block_size=16)
    engine.submit(make_request(32, 4, req_id="r0"))

    engine.run_to_completion()
    summary = engine.metrics.summary(elapsed=engine.clock.now())

    assert 0.0 < summary.peak_kv_utilization <= 1.0
    for step in engine.metrics.steps:
        expected = (step.kv_total_blocks - step.kv_free_blocks) / step.kv_total_blocks
        close(step.kv_utilization, expected)
        assert 0.0 <= step.kv_utilization <= 1.0


def check_batch_size_metrics_track_the_scheduler() -> None:
    engine = build(max_batch_tokens=16)
    for i in range(3):
        engine.submit(make_request(6, 4, req_id=f"r{i}"))

    engine.run_to_completion()
    summary = engine.metrics.summary(elapsed=engine.clock.now())

    assert summary.max_prefill_tokens_per_step <= 16, "the budget is a hard ceiling"
    assert summary.max_prefill_tokens_per_step == 16, (
        "the scheduler packs prompts up to the budget: 6 + 6 + 4"
    )
    assert summary.max_decode_batch == 3, "all three requests end up decoding together"
    assert summary.mean_decode_batch >= 0.0


def check_step_metrics_cover_every_engine_step() -> None:
    engine = build()
    engine.submit(make_request(8, 3, req_id="r0"))
    engine.submit(make_request(12, 2, req_id="r1"))

    engine.run_to_completion()

    assert len(engine.metrics.steps) == len(engine.history)
    for recorded, step in zip(engine.metrics.steps, engine.history, strict=True):
        assert recorded.index == step.index
        close(recorded.duration, step.duration)
        assert recorded.num_scheduled_tokens == step.output.num_scheduled_tokens


def check_preemptions_are_counted_per_request() -> None:
    engine = build(num_blocks=16, block_size=8, enable_preemption=True)
    for i in range(4):
        engine.submit(make_request(8, 32, req_id=f"R{i}"))

    engine.run_to_completion(max_steps=4000)
    summary = engine.metrics.summary(elapsed=engine.clock.now())

    assert summary.num_preemptions > 0, "this workload must force evictions"
    assert summary.num_preemptions == sum(
        r.num_preemptions for r in engine.requests.values()
    )
    for request in engine.requests.values():
        assert (
            engine.metrics.requests[request.id].num_preemptions == request.num_preemptions
        )


def check_percentiles_are_ordered() -> None:
    engine = build()
    for i in range(12):
        engine.submit(make_request(4 + i, 6, req_id=f"r{i}"))

    engine.run_to_completion()
    summary = engine.metrics.summary(elapsed=engine.clock.now())

    assert summary.p50_ttft <= summary.p95_ttft <= summary.p99_ttft
    assert summary.p50_itl <= summary.p95_itl <= summary.p99_itl
    assert summary.mean_ttft >= summary.p50_ttft or summary.p95_ttft >= summary.mean_ttft


def check_explicit_collector_can_be_injected() -> None:
    collector = MetricsCollector()
    engine = Engine(
        EngineConfig(max_batch_tokens=32, num_blocks=32, block_size=16, runner=ZERO_COST),
        metrics=collector,
    )
    engine.submit(make_request(8, 3, req_id="r0"))

    engine.run_to_completion()

    assert engine.metrics is collector
    assert len(collector.steps) == len(engine.history)
    assert isinstance(collector.summary(elapsed=1.0), RunMetrics)


def check_metrics_are_deterministic_under_a_virtual_clock() -> None:
    """Simulated runs must be reproducible, or benchmark comparisons mean nothing."""
    def run() -> dict:
        engine = build()
        for i in range(6):
            engine.submit(make_request(4 + i, 8, arrival=float(i), req_id=f"r{i}"))
        engine.run_to_completion()
        return engine.metrics.summary(elapsed=engine.clock.now()).as_row()

    first, second = run(), run()
    assert first == second, "identical workloads must produce identical metrics"


def main() -> int:
    check("percentile matches numpy", check_percentile_matches_numpy_definition)
    check("percentile validates input", check_percentile_rejects_bad_quantiles)
    check("TTFT is arrival to first token", check_ttft_is_arrival_to_first_token)
    check("TTFT includes queueing delay", check_ttft_includes_queueing_delay)
    check("TPOT matches the published definition", check_tpot_matches_the_published_definition)
    check("TPOT undefined for one token", check_tpot_is_undefined_for_a_single_token)
    check("ITL is the gap between tokens", check_itl_is_the_gap_between_consecutive_tokens)
    check("queue time is arrival to first run", check_queue_time_is_arrival_to_first_run)
    check("E2E covers arrival to completion", check_e2e_covers_arrival_to_completion)
    check("throughput is tokens over elapsed", check_throughput_is_tokens_over_elapsed_time)
    check("generated count matches the engine", check_generated_token_count_matches_the_engine)
    check("KV utilization comes from step state", check_kv_utilization_comes_from_step_state)
    check("batch metrics track the scheduler", check_batch_size_metrics_track_the_scheduler)
    check("step metrics cover every step", check_step_metrics_cover_every_engine_step)
    check("preemptions counted per request", check_preemptions_are_counted_per_request)
    check("percentiles are ordered", check_percentiles_are_ordered)
    check("collector can be injected", check_explicit_collector_can_be_injected)
    check("metrics are deterministic", check_metrics_are_deterministic_under_a_virtual_clock)
    return report("phase 4 metrics")


if __name__ == "__main__":
    raise SystemExit(main())
