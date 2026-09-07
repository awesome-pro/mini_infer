"""Checks for the scheduling policies and the trade-offs between them.

Run with::

    python scripts/check_policies.py
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, expect_raises, report

from mini_infer import Engine, EngineConfig, Request, RunnerConfig
from mini_infer.benchmark.benchmark import BenchmarkCase
from mini_infer.benchmark.driver import Driver
from mini_infer.benchmark.workloads import (
    Constant,
    LengthProfile,
    Uniform,
    WorkloadSpec,
)
from mini_infer.engine.policies import (
    POLICIES,
    SchedulerBase,
    build_policy,
    policy_names,
)
from mini_infer.engine.request import RequestStatus

ZERO_COST = RunnerConfig(
    prefill_base_ms=0.0,
    prefill_ms_per_token=1.0,
    decode_base_ms=0.0,
    decode_ms_per_token=1.0,
    decode_ms_per_context_token=0.0,
)


def workload(
    num_requests: int,
    prompt: int | LengthProfile,
    output: int | LengthProfile,
    **kwargs,
) -> WorkloadSpec:
    return WorkloadSpec(
        num_requests=num_requests,
        prompt=LengthProfile(Constant(prompt)) if isinstance(prompt, int) else prompt,
        output=LengthProfile(Constant(output)) if isinstance(output, int) else output,
        **kwargs,
    )


def engine_for(policy: str, *, runner: RunnerConfig = ZERO_COST, **config) -> Engine:
    return Engine(EngineConfig(policy=policy, runner=runner, **config))


def run(policy: str, spec: WorkloadSpec, *, runner: RunnerConfig = ZERO_COST, **config):
    engine = engine_for(policy, runner=runner, **config)
    return engine, Driver(engine, spec.build()).run()


def fingerprint(engine: Engine) -> list[tuple]:
    """The complete schedule a policy produced: work, evictions and allocations."""
    return [
        (
            step.index,
            tuple((w.request_id, w.num_new_tokens, w.kind.value) for w in step.output.work),
            tuple(sorted(step.preemptions.items())),
            tuple(sorted(step.output.allocations.items())),
        )
        for step in engine.history
    ]


def mixed_spec() -> WorkloadSpec:
    return workload(40, 24, 12, arrival="poisson", rate=30.0, seed=5)


# ------------------------------------------------------------------ registry


def check_every_registered_policy_builds() -> None:
    assert policy_names() == tuple(sorted(policy_names())), "names must be stable"
    for name in policy_names():
        policy = build_policy(name, EngineConfig())
        assert isinstance(policy, SchedulerBase)
        assert policy.name == name


def check_policy_metadata_is_consistent() -> None:
    assert set(POLICIES) == set(policy_names())
    for name, policy_cls in POLICIES.items():
        assert policy_cls.name == name, f"{policy_cls.__name__} name does not match its key"
        assert policy_cls.__doc__, f"{name} must document what it changes"


def check_unknown_policy_is_rejected() -> None:
    def build() -> None:
        build_policy("nope", EngineConfig())

    for name in policy_names():
        expect_raises(ValueError, name, build)
    expect_raises(ValueError, "unknown policy", build)


def check_negative_fairness_knobs_are_rejected() -> None:
    expect_raises(ValueError, "max_wait_steps", lambda: EngineConfig(max_wait_steps=-1))
    expect_raises(ValueError, "prefill_reservation", lambda: EngineConfig(prefill_reservation=-1))
    assert EngineConfig(max_wait_steps=0).max_wait_steps == 0, "0 disables ageing"


# ------------------------------------------------------------------ invariants


def check_token_budget_is_a_hard_ceiling() -> None:
    spec = mixed_spec()
    for policy in policy_names():
        engine, _ = run(
            policy, spec, max_batch_tokens=16, max_running_requests=4, num_blocks=32
        )
        for step in engine.history:
            assert step.output.num_scheduled_tokens <= 16, (
                f"{policy}: {step.output.num_scheduled_tokens} tokens scheduled in step "
                f"{step.index} exceeds the budget"
            )


def check_decode_grants_at_most_one_token() -> None:
    spec = mixed_spec()
    for policy in policy_names():
        engine, _ = run(
            policy, spec, max_batch_tokens=16, max_running_requests=4, num_blocks=32
        )
        for step in engine.history:
            for work in step.output.decode_work:
                assert work.num_new_tokens <= 1, f"{policy}: decode granted {work.num_new_tokens}"


def check_prefill_never_passes_the_prompt() -> None:
    spec = mixed_spec()
    for policy in policy_names():
        engine, _ = run(
            policy, spec, max_batch_tokens=16, max_running_requests=4, num_blocks=32
        )
        for request in engine.requests.values():
            assert request.num_computed_tokens <= request.num_tokens, (
                f"{policy}: {request.id} computed more tokens than it has"
            )
            assert request.num_generated <= request.max_new_tokens
            assert request.num_uncomputed_prompt_tokens >= 0


def check_running_cap_is_respected() -> None:
    spec = mixed_spec()
    for policy in policy_names():
        engine, _ = run(
            policy, spec, max_batch_tokens=16, max_running_requests=4, num_blocks=64
        )
        for step in engine.history:
            scheduled = {w.request_id for w in step.output.work}
            assert len(scheduled) <= 4, f"{policy}: {len(scheduled)} requests in one step"
            assert step.running_requests <= 4, f"{policy}: {step.running_requests} running"


def check_kv_invariants_hold_under_pressure() -> None:
    """Every policy must survive a pool too small for its own running set."""
    spec = workload(6, 8, 12, arrival="burst")
    for policy in policy_names():
        engine, result = run(
            policy,
            spec,
            max_batch_tokens=16,
            max_running_requests=4,
            num_blocks=10,
            block_size=8,
            enable_preemption=True,
        )
        assert engine.memory is not None
        for step in engine.history:
            engine.memory.assert_invariants()
            assert step.free_blocks is not None and step.free_blocks >= 0
        assert result.metrics.num_finished == 6, f"{policy}: only {result.metrics.num_finished}/6"
        assert engine.memory.num_free_blocks() == engine.memory.num_blocks, (
            f"{policy}: leaked KV blocks"
        )
        for request in engine.requests.values():
            assert request.status is RequestStatus.FINISHED
            assert request.block_table.num_blocks == 0, f"{policy}: {request.id} still holds KV"


def check_policies_are_deterministic() -> None:
    spec = mixed_spec()
    for policy in policy_names():
        first = BenchmarkCase(
            label=policy,
            workload=spec,
            config=EngineConfig(
                policy=policy, max_batch_tokens=16, max_running_requests=4,
                num_blocks=32, runner=ZERO_COST,
            ),
        ).run()
        second = BenchmarkCase(
            label=policy,
            workload=spec,
            config=EngineConfig(
                policy=policy, max_batch_tokens=16, max_running_requests=4,
                num_blocks=32, runner=ZERO_COST,
            ),
        ).run()
        assert first.metrics.as_row() == second.metrics.as_row(), (
            f"{policy}: two identical runs disagreed, so benchmarks are noise"
        )


# --------------------------------------------------------------- equivalences


def check_fcfs_is_the_decode_first_schedule() -> None:
    """The default policy is decode priority; that is a finding, not an omission."""
    spec = mixed_spec()
    shared = dict(max_batch_tokens=16, max_running_requests=4, num_blocks=32)
    fcfs, _ = run("fcfs", spec, **shared)
    decode_first, _ = run("decode_first", spec, **shared)
    assert fingerprint(fcfs) == fingerprint(decode_first)


def check_balanced_without_its_knobs_is_decode_first() -> None:
    """With no reservation and no ageing, balanced has nothing left to change."""
    spec = mixed_spec()
    shared = dict(max_batch_tokens=8, max_running_requests=32, num_blocks=64)
    balanced, _ = run("balanced", spec, prefill_reservation=0, max_wait_steps=0, **shared)
    decode_first, _ = run("decode_first", spec, **shared)
    assert fingerprint(balanced) == fingerprint(decode_first)


# --------------------------------------------------------------- experiments


def late_prompt_engine(policy: str) -> tuple[Engine, Request, float, dict[str, float]]:
    """Warm a full decode batch, then submit one long prompt behind it.

    Returns the engine, the late request, its time-to-first-token, and how long each
    warm decoder waited for its next token after the late request arrived.
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
    late_ttft = 0.0
    for _ in range(400):
        step = engine.step()
        for work in step.output.work:
            if work.kind.value == "decode" and work.num_new_tokens and work.request_id != "late":
                warm_next.setdefault(work.request_id, engine.clock.now())
        if late.first_token_time is not None:
            late_ttft = late.first_token_time - submitted_at
            break
    waits = {req_id: when - submitted_at for req_id, when in warm_next.items()}
    return engine, late, late_ttft, waits


def check_prefill_first_trades_itl_for_ttft() -> None:
    """The classic trade-off: prefills first start sooner and stall the decoders."""
    _, _, decode_ttft, decode_waits = late_prompt_engine("decode_first")
    _, _, prefill_ttft, prefill_waits = late_prompt_engine("prefill_first")

    assert decode_waits and prefill_waits, "the warm decoders must produce tokens"
    assert prefill_ttft < decode_ttft * 0.7, (
        f"prefill-first should start the long prompt sooner: {prefill_ttft:.4f} "
        f"vs decode-first {decode_ttft:.4f}"
    )
    assert min(prefill_waits.values()) > min(decode_waits.values()) * 2, (
        "prefill-first must visibly delay the decoders: "
        f"{min(prefill_waits.values()):.4f} vs {min(decode_waits.values()):.4f}"
    )


def fairness(policy: str, spec: WorkloadSpec, **config) -> dict[str, int]:
    """Steps each request waited for its first compute, and how many never got one."""
    engine = engine_for(
        policy, max_batch_tokens=8, max_running_requests=64, num_blocks=1024, **config
    )
    first_seen: dict[str, int] = {}
    first_served: dict[str, int] = {}
    for step in engine.run(spec.build().requests, max_steps=4000):
        for request in engine.waiting:
            first_seen.setdefault(request.id, step.index)
        for work in step.output.work:
            if work.num_new_tokens > 0:
                first_served.setdefault(work.request_id, step.index)
    waits = [first_served[k] - v for k, v in first_seen.items() if k in first_served]
    return {
        "finished": len(engine.finished),
        "max_wait": max(waits) if waits else 0,
        "unserved": sum(1 for k in first_seen if k not in first_served),
    }


def check_ageing_and_the_reservation_both_bound_the_wait() -> None:
    """Decode-first can push one request's first compute hundreds of steps out."""
    spec = workload(40, 8, 40, arrival="burst")
    base = fairness("decode_first", spec)
    assert base["finished"] == 40 and base["unserved"] == 0, base
    assert base["max_wait"] > 150, f"this workload must expose the unfairness: {base}"

    reservation_only = fairness("balanced", spec, prefill_reservation=2, max_wait_steps=0)
    ageing_only = fairness("balanced", spec, prefill_reservation=0, max_wait_steps=8)
    both = fairness("balanced", spec, prefill_reservation=2, max_wait_steps=8)

    variants = (
        ("reservation", reservation_only),
        ("ageing", ageing_only),
        ("both", both),
    )
    for label, result in variants:
        assert result["finished"] == 40, f"{label}: only {result['finished']}/40 finished"
        assert result["unserved"] == 0, f"{label}: a request never got any compute"
        assert result["max_wait"] < base["max_wait"], (
            f"{label} must cut the worst wait: {result['max_wait']} vs {base['max_wait']}"
        )
    # The floor raises the prefill share; ageing is what actually caps the wait,
    # because it decides which request the floor is spent on.
    assert ageing_only["max_wait"] < reservation_only["max_wait"], (
        f"ageing must beat a floor alone: {ageing_only['max_wait']} vs "
        f"{reservation_only['max_wait']}"
    )
    assert both["max_wait"] <= ageing_only["max_wait"] + 1, (
        "adding the reservation must not loosen the ageing bound: "
        f"{both['max_wait']} vs {ageing_only['max_wait']}"
    )


def admission_waves(engine: Engine) -> list[tuple[int, set[str], set[int]]]:
    """Group requests by the step they were admitted, with the steps they ran in."""
    active: dict[str, set[int]] = {}
    for step in engine.history:
        for work in step.output.work:
            active.setdefault(work.request_id, set()).add(step.index)
    waves: dict[int, tuple[set[str], set[int]]] = {}
    for request in engine.requests.values():
        ids, steps = waves.setdefault(request.admitted_step, (set(), set()))
        ids.add(request.id)
        steps.update(active.get(request.id, set()))
    return [(step, ids, steps) for step, (ids, steps) in sorted(waves.items())]


def check_static_batching_never_refills_mid_batch() -> None:
    """A batch is formed once, and the next batch starts only after it drains."""
    spec = workload(16, 8, Uniform(4, 64), arrival="burst", seed=11)
    engine, result = run(
        "static", spec, max_batch_tokens=32, max_running_requests=4, num_blocks=256
    )
    assert result.metrics.num_finished == 16

    waves = admission_waves(engine)
    assert len(waves) >= 2, "a static run should form several batches"
    for (_, _, earlier), (_, _, later) in pairwise(waves):
        assert max(earlier) < min(later), (
            "a new request joined while the previous batch was still running"
        )


def check_static_batching_costs_latency() -> None:
    """Freezing the batch leaves slots idle, which shows up as latency and throughput."""
    spec = workload(16, 8, Uniform(4, 64), arrival="burst", seed=11)
    shared = dict(max_batch_tokens=32, max_running_requests=4, num_blocks=256)
    _, continuous = run("fcfs", spec, **shared)
    _, static = run("static", spec, **shared)

    assert continuous.metrics.num_finished == static.metrics.num_finished == 16
    assert static.metrics.mean_e2e > continuous.metrics.mean_e2e, (
        "static batching must pay in end-to-end latency: "
        f"{static.metrics.mean_e2e:.4f} vs {continuous.metrics.mean_e2e:.4f}"
    )
    assert static.metrics.output_throughput < continuous.metrics.output_throughput, (
        "static batching must pay in throughput: "
        f"{static.metrics.output_throughput:.1f} vs {continuous.metrics.output_throughput:.1f}"
    )


def check_identical_lengths_hide_static_batching() -> None:
    """A batch whose members finish together drains as fast as it refills."""
    spec = workload(16, 8, 32, arrival="burst")
    shared = dict(max_batch_tokens=32, max_running_requests=4, num_blocks=256)
    _, continuous = run("fcfs", spec, **shared)
    _, static = run("static", spec, **shared)

    assert static.metrics.mean_e2e == continuous.metrics.mean_e2e, (
        "with identical lengths the whole batch finishes at once, so freezing costs "
        "nothing — this is why the static-batching comparison needs mixed lengths"
    )


def main() -> int:
    check("every registered policy builds", check_every_registered_policy_builds)
    check("policy metadata is consistent", check_policy_metadata_is_consistent)
    check("unknown policy is rejected", check_unknown_policy_is_rejected)
    check("negative fairness knobs are rejected", check_negative_fairness_knobs_are_rejected)
    check("token budget is a hard ceiling", check_token_budget_is_a_hard_ceiling)
    check("decode grants at most one token", check_decode_grants_at_most_one_token)
    check("prefill never passes the prompt", check_prefill_never_passes_the_prompt)
    check("running cap is respected", check_running_cap_is_respected)
    check("KV invariants hold under pressure", check_kv_invariants_hold_under_pressure)
    check("policies are deterministic", check_policies_are_deterministic)
    check("fcfs is the decode-first schedule", check_fcfs_is_the_decode_first_schedule)
    check(
        "balanced without its knobs is decode-first",
        check_balanced_without_its_knobs_is_decode_first,
    )
    check("prefill-first trades ITL for TTFT", check_prefill_first_trades_itl_for_ttft)
    check(
        "ageing and the reservation bound the wait",
        check_ageing_and_the_reservation_both_bound_the_wait,
    )
    check("static batching never refills mid-batch", check_static_batching_never_refills_mid_batch)
    check("static batching costs latency", check_static_batching_costs_latency)
    check("identical lengths hide static batching", check_identical_lengths_hide_static_batching)
    return report("phase 6 scheduling policies")


if __name__ == "__main__":
    raise SystemExit(main())
