"""Checks for the Phase 1 skeleton: request model, budget, continuous batching.

Run with::

    python scripts/check_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, close, expect_raises, report  # noqa: E402

from mini_infer.clock import Clock, VirtualClock, WallClock  # noqa: E402
from mini_infer.config import EngineConfig, RunnerConfig  # noqa: E402
from mini_infer.engine.engine import Engine, StepEventKind  # noqa: E402
from mini_infer.engine.request import Request, RequestStatus  # noqa: E402
from mini_infer.engine.scheduler import (  # noqa: E402
    ScheduledKind,
    SchedulerOutput,
    ScheduledWork,
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


def tiny_config(**overrides) -> EngineConfig:
    base: dict = {
        "max_batch_tokens": 16,
        "max_running_requests": 4,
        "num_blocks": 64,
        "block_size": 16,
        "enable_chunked_prefill": False,
        "runner": RunnerConfig(
            prefill_base_ms=0.0,
            prefill_ms_per_token=1.0,
            decode_base_ms=0.0,
            decode_ms_per_token=1.0,
            decode_ms_per_context_token=0.0,
        ),
    }
    base.update(overrides)
    return EngineConfig(**base)


class NullScheduler:
    def schedule(self, waiting, running, *, now, budget) -> SchedulerOutput:
        return SchedulerOutput(work=())


# --------------------------------------------------------------------- request


def check_request_lifecycle() -> None:
    req = make_request(10, 4)
    assert req.status is RequestStatus.WAITING
    assert req.remaining_prefill == 10
    assert req.sequence_len == 0

    req.on_prefill(4)
    assert req.status is RequestStatus.PREFILLING
    assert req.prefilled_tokens == 4

    req.on_prefill(6)
    assert req.status is RequestStatus.DECODING
    assert req.remaining_prefill == 0

    req.on_decode(token=7, now=0.5)
    assert req.first_token_time == 0.5
    assert req.sequence_len == 11

    req.on_finished(now=0.9)
    assert req.status is RequestStatus.FINISHED
    assert req.finish_time == 0.9


def check_request_validation() -> None:
    expect_raises(
        ValueError, "prompt_tokens", lambda: Request(prompt_tokens=[], max_new_tokens=1, arrival_time=0.0)
    )
    expect_raises(ValueError, "max_new_tokens", lambda: make_request(4, -1))

    req = make_request(5, 1)
    expect_raises(ValueError, "exceed prompt", lambda: req.on_prefill(6))
    expect_raises(ValueError, "cannot decode", lambda: req.on_decode(token=1, now=0.0))


def check_preemption_rewinds_prefill() -> None:
    req = make_request(6, 2)
    req.on_prefill(6)
    req.on_decode(token=1, now=0.1)
    req.block_table = [3, 9]

    req.on_preempted()

    assert req.status is RequestStatus.PREEMPTED
    assert req.prefilled_tokens == 0
    assert req.block_table == []
    assert req.num_preemptions == 1
    assert req.generated_tokens == [1], "generated tokens survive preemption (recompute path)"

    req.on_admitted(step=7)
    req.on_prefill(6)
    assert req.status is RequestStatus.DECODING
    assert req.sequence_len == 7, "recompute must restore prompt KV plus prior output"


def check_scheduled_work_validation() -> None:
    req = make_request(4, 1)
    expect_raises(
        ValueError,
        "non-negative",
        lambda: ScheduledWork(request=req, num_new_tokens=-1, kind=ScheduledKind.PREFILL),
    )
    expect_raises(
        ValueError,
        "exceeds",
        lambda: ScheduledWork(request=req, num_new_tokens=5, kind=ScheduledKind.PREFILL),
    )
    zero = ScheduledWork(request=req, num_new_tokens=0, kind=ScheduledKind.PREFILL)
    assert zero.num_new_tokens == 0, "a zero-token prefill is how a slot is claimed"
    req.on_prefill(4)
    expect_raises(
        ValueError,
        "exactly one token",
        lambda: ScheduledWork(request=req, num_new_tokens=3, kind=ScheduledKind.DECODE),
    )


def check_scheduler_output_aggregation() -> None:
    prefill_req = make_request(10, 1)
    decode_req = make_request(3, 1)
    decode_req.on_prefill(3)
    output = SchedulerOutput(
        work=(
            ScheduledWork(request=decode_req, num_new_tokens=1, kind=ScheduledKind.DECODE),
            ScheduledWork(request=prefill_req, num_new_tokens=6, kind=ScheduledKind.PREFILL),
        )
    )
    assert output.num_scheduled_tokens == 7
    assert output.num_prefill_tokens == 6
    assert output.num_decode_requests == 1
    assert output.tokens_for(decode_req.id) == 1
    assert output.tokens_for("missing") == 0
    assert prefill_req.id in output
    assert not output.is_empty
    assert SchedulerOutput(work=()).is_empty


# ---------------------------------------------------------------------- config


def check_config_validation() -> None:
    for field in (
        "max_batch_tokens",
        "max_running_requests",
        "num_blocks",
        "block_size",
        "max_prefill_chunk",
        "dtype_bytes",
    ):
        expect_raises(ValueError, field, lambda f=field: EngineConfig(**{f: 0}))
    expect_raises(ValueError, "prefill_ms_per_token", lambda: RunnerConfig(prefill_ms_per_token=-1.0))

    config = EngineConfig()
    assert config.kv_capacity_tokens == config.num_blocks * config.block_size
    assert config.with_(block_size=32).block_size == 32
    assert config.block_size == 16


def check_clocks() -> None:
    assert isinstance(VirtualClock(), Clock)
    assert isinstance(WallClock(), Clock)

    clock = VirtualClock(start=2.0)
    clock.advance(0.25)
    assert clock.now() == 2.25
    expect_raises(ValueError, "backwards", lambda: clock.advance(-0.1))


# ---------------------------------------------------------------------- engine


def check_single_request_runs_to_completion() -> None:
    engine = Engine(tiny_config())
    req = engine.submit(make_request(4, 3))

    steps = engine.run_to_completion()

    assert [s.num_prefill_tokens for s in steps] == [4, 0, 0, 0]
    assert [s.num_decode_requests for s in steps] == [0, 1, 1, 1]
    assert req.status is RequestStatus.FINISHED
    assert req.num_generated == 3
    assert engine.finished == [req]
    assert engine.running == {}


def check_budget_always_wins_over_prompt_size() -> None:
    engine = Engine(tiny_config(max_batch_tokens=8))
    engine.submit(make_request(30, 1))

    steps = engine.run_to_completion()

    prefill_chunks = [s.num_prefill_tokens for s in steps if s.num_prefill_tokens]

    assert prefill_chunks == [8, 8, 8, 6], (
        "a prompt larger than the budget must be split, never stalled"
    )


def check_chunking_off_prefills_in_budget_sized_bites() -> None:
    config = tiny_config(max_batch_tokens=30, enable_chunked_prefill=False)
    engine = Engine(config)
    engine.submit(make_request(20, 1))

    steps = engine.run_to_completion()

    assert steps[0].num_prefill_tokens == 20, "one shot when it fits the step"


def check_chunking_on_respects_the_chunk_cap() -> None:
    config = tiny_config(max_batch_tokens=30, enable_chunked_prefill=True, max_prefill_chunk=16)
    engine = Engine(config)
    engine.submit(make_request(20, 1))

    first = engine.step()

    assert first.num_prefill_tokens == 16, "the configured chunk cap must bind"


def check_a_giant_prompt_does_not_starve_forever() -> None:
    config = tiny_config(max_batch_tokens=8, enable_chunked_prefill=False)
    engine = Engine(config)
    req = engine.submit(make_request(30, 1))

    steps = engine.run_to_completion()

    assert req.status is RequestStatus.FINISHED
    assert sum(s.num_prefill_tokens for s in steps) == 30


def check_token_budget_never_exceeded() -> None:
    config = tiny_config(max_batch_tokens=10, max_running_requests=8)
    engine = Engine(config)
    for _ in range(3):
        engine.submit(make_request(6, 2))

    for step in engine.run():
        assert step.output.num_scheduled_tokens <= config.max_batch_tokens


def check_running_limit_respected() -> None:
    config = tiny_config(max_batch_tokens=64, max_running_requests=2)
    engine = Engine(config)
    for _ in range(5):
        engine.submit(make_request(4, 3))

    peak = 0
    for step in engine.run():
        active = {w.request_id for w in step.output.work}
        peak = max(peak, len(active))
        assert len(active) <= config.max_running_requests
    assert peak == 2


def check_continuous_batching_reuses_slots() -> None:
    config = tiny_config(max_batch_tokens=64, max_running_requests=1)
    engine = Engine(config)
    short = engine.submit(make_request(2, 1))
    long = engine.submit(make_request(2, 10))

    steps = engine.run_to_completion()

    short_last = max(i for i, s in enumerate(steps) if short.id in s.output)
    long_first = next(i for i, s in enumerate(steps) if long.id in s.output)
    assert short_last < long_first, "the freed slot must be handed to the waiting request"
    assert long.status is RequestStatus.FINISHED


def check_decode_and_prefill_share_a_step() -> None:
    engine = Engine(tiny_config(max_batch_tokens=16))
    first = engine.submit(make_request(4, 3))
    list(engine.run(max_steps=1))  # advance one step so `first` is decoding
    second = engine.submit(make_request(4, 2))

    step = next(engine.run())

    assert step.num_decode_requests == 1
    assert step.num_prefill_tokens == 4
    assert {w.request_id for w in step.output.work} == {first.id, second.id}


def check_arrivals_and_idle_skip() -> None:
    engine = Engine(tiny_config())
    req = make_request(2, 1, arrival=5.0)
    engine.submit_at(req)

    steps = engine.run_to_completion()

    assert steps[0].start_time == 5.0, "engine must jump the clock to the next arrival"
    assert req.arrival_time == 5.0


def check_past_arrival_is_clamped() -> None:
    engine = Engine(tiny_config(), clock=VirtualClock(start=3.0))
    req = engine.submit(make_request(2, 1, arrival=0.5))
    assert req.arrival_time == 3.0


def check_duplicate_ids_rejected() -> None:
    engine = Engine(tiny_config())
    engine.submit(make_request(2, 1, req_id="dup"))
    expect_raises(ValueError, "duplicate request id", lambda: engine.submit(make_request(2, 1, req_id="dup")))
    expect_raises(
        ValueError,
        "duplicate request id",
        lambda: engine.submit_at(make_request(2, 1, arrival=9.0, req_id="dup")),
    )


def check_event_lifecycle() -> None:
    engine = Engine(tiny_config())
    engine.submit(make_request(4, 1))

    steps = engine.run_to_completion()

    assert [e.kind for e in steps[0].events] == [StepEventKind.ADMITTED, StepEventKind.PREFILLED]
    assert [e.kind for e in steps[1].events] == [StepEventKind.DECODED, StepEventKind.FINISHED]
    assert steps[0].events_of(StepEventKind.PREFILLED)[0].num_tokens == 4


def check_zero_output_budget_still_completes() -> None:
    """A request always decodes at least once, then the budget stops it."""
    engine = Engine(tiny_config())
    req = engine.submit(make_request(4, 0))

    steps = engine.run_to_completion()

    assert steps[0].num_prefill_tokens == 4
    assert req.status is RequestStatus.FINISHED
    assert req.generated_tokens == [0]
    assert engine.finished == [req]


def check_idle_engine_terminates() -> None:
    engine = Engine(tiny_config(), scheduler=NullScheduler())
    engine.submit(make_request(2, 1))

    assert engine.run_to_completion() == []
    assert engine.waiting, "unscheduled work must remain queued"


def check_step_duration_matches_cost_model() -> None:
    config = tiny_config(
        runner=RunnerConfig(
            prefill_base_ms=0.0,
            prefill_ms_per_token=2.0,
            decode_base_ms=0.0,
            decode_ms_per_token=1.0,
            decode_ms_per_context_token=0.0,
        )
    )
    engine = Engine(config)
    engine.submit(make_request(4, 1))

    steps = engine.run_to_completion()
    prefill_step = next(s for s in steps if s.num_prefill_tokens)
    decode_step = next(s for s in steps if s.num_decode_requests)

    close(prefill_step.duration, 0.008)  # 4 prefill tokens at 2ms
    close(decode_step.duration, 0.001)  # 1 decode token at 1ms
    close(decode_step.end_time - prefill_step.end_time, 0.001)  # clock accumulates durations


def check_decode_cost_scales_with_context() -> None:
    config = tiny_config(
        runner=RunnerConfig(
            prefill_base_ms=0.0,
            prefill_ms_per_token=0.0,
            decode_base_ms=0.0,
            decode_ms_per_token=0.0,
            decode_ms_per_context_token=1.0,
        )
    )
    engine = Engine(config)
    req = engine.submit(make_request(4, 1))

    steps = engine.run_to_completion()
    decode_step = next(s for s in steps if s.num_decode_requests)

    assert decode_step.duration == 0.005, "cost is paid over the full context length"
    assert req.sequence_len == 5


def main() -> int:
    check("request lifecycle", check_request_lifecycle)
    check("request validation", check_request_validation)
    check("preemption rewinds prefill", check_preemption_rewinds_prefill)
    check("scheduled work validation", check_scheduled_work_validation)
    check("scheduler output aggregation", check_scheduler_output_aggregation)
    check("config validation", check_config_validation)
    check("clock protocols", check_clocks)
    check("single request to completion", check_single_request_runs_to_completion)
    check("budget wins over prompt size", check_budget_always_wins_over_prompt_size)
    check("unchunked prefill is one shot when it fits", check_chunking_off_prefills_in_budget_sized_bites)
    check("chunk cap binds when chunking is on", check_chunking_on_respects_the_chunk_cap)
    check("a giant prompt never starves", check_a_giant_prompt_does_not_starve_forever)
    check("token budget never exceeded", check_token_budget_never_exceeded)
    check("running request limit", check_running_limit_respected)
    check("continuous batching slot reuse", check_continuous_batching_reuses_slots)
    check("decode and prefill share a step", check_decode_and_prefill_share_a_step)
    check("arrivals and idle skip", check_arrivals_and_idle_skip)
    check("past arrival clamped", check_past_arrival_is_clamped)
    check("duplicate ids rejected", check_duplicate_ids_rejected)
    check("event lifecycle", check_event_lifecycle)
    check("zero output budget still completes", check_zero_output_budget_still_completes)
    check("idle engine terminates", check_idle_engine_terminates)
    check("step duration matches cost model", check_step_duration_matches_cost_model)
    check("decode cost scales with context", check_decode_cost_scales_with_context)
    return report("phase 1 skeleton")


if __name__ == "__main__":
    raise SystemExit(main())
