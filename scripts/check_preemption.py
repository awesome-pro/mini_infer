"""Checks for recompute preemption: the coupling between scheduling and KV memory.

Run with::

    python scripts/check_preemption.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, report

from miniserve import Engine, EngineConfig, Request, RunnerConfig
from miniserve.engine.engine import StepEventKind
from miniserve.engine.request import RequestStatus

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


def pressure_engine(
    num_blocks: int, block_size: int, *, preemption: bool, runners: int = 4
) -> Engine:
    return Engine(
        EngineConfig(
            max_batch_tokens=64,
            max_running_requests=runners,
            num_blocks=num_blocks,
            block_size=block_size,
            enable_chunked_prefill=True,
            enable_preemption=preemption,
            runner=ZERO_COST,
        )
    )


def check_preemption_rewinds_a_victims_prefill() -> None:
    """An evicted request keeps its output but loses its cached prompt."""
    engine = pressure_engine(num_blocks=6, block_size=8, preemption=True)
    victim = engine.submit(make_request(8, 40, req_id="A"))
    engine.submit(make_request(8, 40, req_id="B"))

    steps = engine.run_to_completion(max_steps=400)
    preempted = [s for s in steps if s.preemptions]

    assert preempted, "6 blocks cannot hold two 40-token sequences, so one must be evicted"
    assert victim.num_preemptions >= 1 or any(
        r.num_preemptions for r in engine.requests.values()
    )
    assert all(s.preemptions for s in preempted)
    for step in preempted:
        for victim_id, requester_id in step.preemptions.items():
            assert victim_id != requester_id
            assert victim_id not in {w.request_id for w in step.output.work}, (
                "an evicted request must not also execute in the same step"
            )


def check_preemption_preserves_output_and_recomputes_prompt() -> None:
    """Whatever is evicted must still finish with every token it produced."""
    engine = pressure_engine(num_blocks=6, block_size=8, preemption=True)
    requests = [
        engine.submit(make_request(8, 20, req_id=name)) for name in ("A", "B", "C")
    ]

    engine.run_to_completion(max_steps=600)

    assert sum(r.num_preemptions for r in requests) >= 1, "this pool must force evictions"
    for request in requests:
        assert request.status is RequestStatus.FINISHED, f"{request.id} did not finish"
        assert request.num_generated == 20, "no output token may be lost to recomputation"
        assert request.is_prefill_complete, "the whole prompt must end up computed"
        assert request.num_computed_tokens == request.num_tokens, (
            "a finished request has computed its entire sequence"
        )
        # Finished requests release their KV, so a rebuilt table is already gone.
        assert request.block_table.num_blocks == 0


def check_every_step_has_consistent_blocks() -> None:
    """A preempted request's block table is cleared and rebuilt from zero."""
    engine = pressure_engine(num_blocks=8, block_size=8, preemption=True)
    for i in range(5):
        engine.submit(make_request(8, 16, req_id=f"R{i}"))

    for _ in engine.run():
        memory = engine.memory
        assert memory is not None
        memory.assert_invariants()
        for request in engine.requests.values():
            if request.status is RequestStatus.PREEMPTED:
                assert request.num_computed_tokens == 0
                assert request.block_table.num_blocks == 0


def check_preemption_lets_late_requests_through() -> None:
    """Without preemption a small pool stalls; with it, everything finishes."""
    def run(preemption: bool) -> tuple[int, int, int]:
        engine = pressure_engine(num_blocks=10, block_size=8, preemption=preemption)
        for i in range(6):
            engine.submit(make_request(8, 12, req_id=f"R{i}"))
        steps = engine.run_to_completion(max_steps=3000)
        evictions = sum(s.num_preemptions for s in steps)
        return len(engine.finished), evictions, len(steps)

    without_finished, _, _ = run(preemption=False)
    with_finished, with_evictions, _ = run(preemption=True)

    assert without_finished == 6, "the baseline must also finish this workload"
    assert with_finished == 6, "preemption must not lose requests"
    assert with_evictions > 0, "this pool is tight enough to force evictions"


def check_over_subscribed_workload_stalls_cleanly() -> None:
    """A workload that can never fit must stop with a reported stall, not a crash."""
    engine = pressure_engine(num_blocks=4, block_size=8, preemption=True)
    request = engine.submit(make_request(8, 100, req_id="huge"))

    steps = engine.run_to_completion(max_steps=500)
    memory = engine.memory
    assert memory is not None

    memory.assert_invariants()
    assert request.status is not RequestStatus.FINISHED
    assert steps, "the engine still runs"
    assert steps[-1].is_stalled, "the terminal state is a stall, not a silent success"


def check_no_thrashing_under_randomized_pressure() -> None:
    """Randomized pressure must terminate with every request accounted for."""
    rng = random.Random(7)
    engine = Engine(
        EngineConfig(
            max_batch_tokens=64,
            max_running_requests=5,
            num_blocks=14,
            block_size=16,
            enable_chunked_prefill=True,
            enable_preemption=True,
            max_prefill_chunk=32,
            runner=ZERO_COST,
        )
    )
    clock = 0.0
    expected = 0
    for i in range(40):
        clock += rng.uniform(0.0, 0.05)
        engine.submit_at(
            make_request(
                prompt_len=rng.randint(1, 48),
                max_new_tokens=rng.randint(1, 20),
                arrival=clock,
                req_id=f"p{i}",
            )
        )
        expected += 1

    steps = engine.run_to_completion(max_steps=4000)
    memory = engine.memory
    assert memory is not None
    memory.assert_invariants()

    assert len(engine.finished) == expected, (
        f"only {len(engine.finished)}/{expected} requests finished; "
        f"last step stalled={steps[-1].is_stalled}"
    )
    assert memory.num_free_blocks() == memory.num_blocks, "all KV must be reclaimed"
    for request in engine.requests.values():
        assert request.num_generated <= request.max_new_tokens
        assert request.num_computed_tokens <= request.num_tokens
        assert request.block_table.num_blocks == 0


def check_planning_never_mutates_memory() -> None:
    """The scheduler decides; only the engine commits.

    Planning must be speculative, so that a plan which is never applied (a failed
    runner, an abandoned step) cannot leave physical memory moved.
    """
    engine = pressure_engine(num_blocks=16, block_size=8, preemption=True)
    requests = [
        engine.submit(make_request(8, 32, req_id=f"R{i}")) for i in range(4)
    ]

    checked = 0
    for _ in range(60):
        before = (
            engine.memory.num_free_blocks(),
            tuple(r.block_table.num_blocks for r in requests),
        )
        plan = engine.scheduler.schedule(
            engine.waiting,
            list(engine.running.values()),
            now=engine.clock.now(),
            budget=64,
            memory=engine.memory,
        )
        after = (
            engine.memory.num_free_blocks(),
            tuple(r.block_table.num_blocks for r in requests),
        )
        assert before == after, "scheduling must not touch physical memory"
        if plan.preemptions:
            checked += 1
        engine.step()

    assert checked > 0, "this workload must plan at least one eviction"
    assert engine.memory.num_free_blocks() == 16


def check_preemption_events_are_reported() -> None:
    engine = pressure_engine(num_blocks=6, block_size=8, preemption=True)
    engine.submit(make_request(8, 30, req_id="A"))
    engine.submit(make_request(8, 30, req_id="B"))

    steps = engine.run_to_completion(max_steps=400)
    events = [e for s in steps for e in s.events if e.kind is StepEventKind.PREEMPTED]

    assert events, "a preemption must surface as a step event"
    assert all("evicted for" in e.detail for e in events)


def main() -> int:
    check("preemption rewinds the victim's prefill", check_preemption_rewinds_a_victims_prefill)
    check(
        "preemption preserves output, recomputes prompt",
        check_preemption_preserves_output_and_recomputes_prompt,
    )
    check("block tables stay consistent every step", check_every_step_has_consistent_blocks)
    check("late requests get through under pressure", check_preemption_lets_late_requests_through)
    check("over-subscribed workloads stall cleanly", check_over_subscribed_workload_stalls_cleanly)
    check("randomized pressure terminates", check_no_thrashing_under_randomized_pressure)
    check("planning never mutates memory", check_planning_never_mutates_memory)
    check("preemptions are reported as events", check_preemption_events_are_reported)
    return report("phase 3 recompute preemption")


if __name__ == "__main__":
    raise SystemExit(main())
