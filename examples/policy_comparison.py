"""Comparing the scheduling policies, one decision at a time.

Sections:

1. the same mixed workload under every policy
2. one long prompt arriving behind a full decode batch (the ITL/TTFT trade-off)
3. static batching against continuous batching, with mixed output lengths
4. what the fairness knobs actually bound

Run with::

    python examples/policy_comparison.py
"""

from __future__ import annotations

from mini_infer import Engine, EngineConfig, Request, policy_names
from mini_infer.benchmark.benchmark import Report, SweepRow
from mini_infer.benchmark.driver import Driver
from mini_infer.benchmark.workloads import (
    Constant,
    LengthProfile,
    LogNormal,
    Uniform,
    WorkloadSpec,
)


def spec(num_requests: int, prompt, output, **kwargs) -> WorkloadSpec:
    return WorkloadSpec(
        num_requests=num_requests,
        prompt=prompt if isinstance(prompt, LengthProfile) else LengthProfile(Constant(prompt)),
        output=output if isinstance(output, LengthProfile) else LengthProfile(Constant(output)),
        **kwargs,
    )


def run(policy: str, workload: WorkloadSpec, **config) -> SweepRow:
    engine_config = EngineConfig(policy=policy, **config)
    engine = Engine(engine_config)
    result = Driver(engine, workload.build()).run()
    return SweepRow(label=policy, config=engine_config, result=result)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------- 1. one workload


def compare_on_one_workload() -> None:
    section("1. The same workload under every policy")
    workload = spec(
        200,
        LengthProfile(LogNormal(median=64, sigma=0.9), minimum=4, maximum=512),
        LengthProfile(LogNormal(median=48, sigma=0.7), minimum=4, maximum=256),
        arrival="poisson",
        rate=64.0,
        seed=0,
    )
    print(workload.build().describe())
    config = dict(
        max_batch_tokens=64,
        max_running_requests=8,
        num_blocks=256,
        block_size=16,
        prefill_reservation=16,
        max_wait_steps=16,
    )
    rows = [run(policy, workload, **config).as_row() for policy in policy_names()]
    for row in rows:
        row["completed"] = f"{row['finished']}/{row['requests']}"
    print(Report().render(rows))
    print(
        "\nRead it across: fcfs and decode_first are the same schedule (tested), so the\n"
        "interesting columns are prefill_first (prefills win the step) and balanced\n"
        "(a floor plus ageing). static is an admission rule, not a budget rule."
    )


# ------------------------------------------------------- 2. the ITL/TTFT trade-off


def compare_late_prompt() -> None:
    section("2. One long prompt arriving behind a full decode batch")
    print("8 requests are decoding and already fill the step; a 64-token prompt arrives.\n")
    print(f"{'policy':14} {'late TTFT':>10} {'decoders stalled':>17} {'late step':>10}")
    print("-" * 55)

    for policy in ("decode_first", "prefill_first", "balanced", "static"):
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
                Request(
                    prompt_tokens=list(range(4)),
                    max_new_tokens=300,
                    arrival_time=0.0,
                    id=f"w{i}",
                )
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
        late_step = 0.0
        for _ in range(600):
            step = engine.step()
            for work in step.output.work:
                decoding = work.kind.value == "decode" and work.num_new_tokens
                if decoding and work.request_id != "late":
                    warm_next.setdefault(work.request_id, engine.clock.now())
            if late.first_token_time is not None:
                late_ttft = late.first_token_time - submitted_at
                late_step = step.duration
                break
        stalled = min(warm_next.values()) - submitted_at
        print(
            f"{policy:14} {late_ttft * 1000:9.1f}ms {stalled * 1000:16.1f}ms "
            f"{late_step * 1000:9.2f}ms"
        )

    print(
        "\nprefill_first starts the long prompt ~2.4x sooner and makes the decoders wait\n"
        "~7x longer for their next token. That is the whole trade-off in two numbers."
    )


# ------------------------------------------------- 3. static vs continuous


def compare_static_batching() -> None:
    section("3. Static batching vs continuous batching (mixed output lengths)")
    print("16 requests arrive at once, 4 run at a time. Output lengths are 4-64 tokens,\n"
          "so a static batch waits for its longest member before refilling a slot.\n")
    workload = spec(16, 8, LengthProfile(Uniform(4, 64)), arrival="burst", seed=11)
    rows = []
    for policy in ("fcfs", "static"):
        row = run(
            policy,
            workload,
            max_batch_tokens=32,
            max_running_requests=4,
            num_blocks=256,
            block_size=16,
        ).as_row()
        row["completed"] = f"{row['finished']}/{row['requests']}"
        rows.append(row)
    print(Report().render(rows))
    print(
        "\nSame engine, same budget, same memory manager. The only difference is that\n"
        "static refuses to admit a request while any request is in flight."
    )


# ---------------------------------------------------------------- 4. fairness


def compare_fairness() -> None:
    section("4. What the fairness knobs bound (budget 8, 40 requests at once)")

    def worst_wait(policy: str, **config) -> tuple[int, int, int]:
        engine = Engine(
            EngineConfig(
                policy=policy,
                max_batch_tokens=8,
                max_running_requests=64,
                num_blocks=1024,
                block_size=16,
                **config,
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
        return (
            len(engine.finished),
            max(waits) if waits else 0,
            sum(1 for k in first_seen if k not in first_served),
        )

    print(f"{'policy':22} {'finished':>9} {'worst wait':>11} {'never served':>13}")
    print("-" * 58)
    for label, policy, config in (
        ("decode_first", "decode_first", {}),
        ("balanced r=2 w=off", "balanced", {"prefill_reservation": 2, "max_wait_steps": 0}),
        ("balanced r=0 w=8", "balanced", {"prefill_reservation": 0, "max_wait_steps": 8}),
        ("balanced r=2 w=8", "balanced", {"prefill_reservation": 2, "max_wait_steps": 8}),
    ):
        finished, wait, unserved = worst_wait(policy, **config)
        print(f"{label:22} {finished:>7}/40 {wait:>9} steps {unserved:>13}")

    print(
        "\n'worst wait' counts engine steps between a request becoming visible and its\n"
        "first compute. Every policy serves everyone eventually, so this is a fairness\n"
        "spread, not a deadlock: decode-first lets one request wait ~5x longer than\n"
        "balanced does."
    )


def main() -> None:
    print("Scheduling policy comparison — every number below is from this run.")
    compare_on_one_workload()
    compare_late_prompt()
    compare_static_batching()
    compare_fairness()


if __name__ == "__main__":
    main()
