"""Step-by-step demo of the Phase 1 engine.

Reproduces the request scenario from the project brief and prints the engine
timeline, so the scheduler's decisions can be read directly.

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, EngineConfig, Request, RunnerConfig  # noqa: E402
from mini_infer.engine.engine import EngineStep, StepEventKind  # noqa: E402

PROMPTS = {
    "R1": 80,
    "R2": 12,
    "R3": 300,
    "R4": 8,
}

ARRIVALS = {"R1": 0.0, "R2": 0.0, "R3": 1.0, "R4": 3.0}
OUTPUTS = {"R1": 20, "R2": 30, "R3": 10, "R4": 50}


def build_engine() -> Engine:
    config = EngineConfig(
        max_batch_tokens=64,
        max_running_requests=4,
        num_blocks=32,
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
    return Engine(config)


def make_requests() -> list[Request]:
    return [
        Request(
            id=name,
            prompt_tokens=list(range(PROMPTS[name])),
            max_new_tokens=OUTPUTS[name],
            arrival_time=ARRIVALS[name],
        )
        for name in ("R1", "R2", "R3", "R4")
    ]


def cell(step: EngineStep, request_id: str) -> str:
    for work in step.output.work:
        if work.request_id == request_id:
            tag = "P" if work.kind.value == "prefill" else "D"
            return f"{tag}{work.num_new_tokens}"
    return "-"


def print_timeline(engine: Engine, requests: list[Request], steps: list[EngineStep]) -> None:
    ids = [r.id for r in requests]
    width = max(len(i) for i in ids) + 2
    print("\n" + "=" * 78)
    print("ENGINE TIMELINE   (P<n> = n prefill tokens, D1 = one decode token)")
    print("=" * 78)

    header = f"{'time':>8} " + "".join(f"{i:>{width}}" for i in ids) + "   batch"
    print(header)
    print("-" * len(header))
    for step in steps:
        row = f"{step.end_time:>8.3f} " + "".join(f"{cell(step, i):>{width}}" for i in ids)
        row += f"   {step.output.num_scheduled_tokens:>3} tok  {step.duration * 1000:>7.2f} ms"
        print(row)

    print("-" * len(header))
    totals = "".join(f"{r.num_generated:>{width}}" for r in requests)
    print(f"{'tokens':>8} " + totals + "   generated per request")


def print_summary(engine: Engine, requests: list[Request], steps: list[EngineStep]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"engine steps        : {len(steps)}")
    print(f"simulated wall time : {engine.clock.now():.3f} s")
    print(f"prefill tokens      : {sum(s.num_prefill_tokens for s in steps)}")
    print(f"decode tokens       : {sum(s.num_decode_requests for s in steps)}")
    print(f"admissions          : {sum(len(s.events_of(StepEventKind.ADMITTED)) for s in steps)}")

    print(f"\n{'req':<5}{'prompt':>7}{'out':>5}{'ttft(ms)':>10}{'e2e(ms)':>10}{'status':>10}")
    for req in requests:
        ttft = (
            (req.first_token_time - req.arrival_time) * 1000
            if req.first_token_time is not None
            else float("nan")
        )
        e2e = (
            (req.finish_time - req.arrival_time) * 1000
            if req.finish_time is not None
            else float("nan")
        )
        print(
            f"{req.id:<5}{req.prompt_len:>7}{req.num_generated:>5}"
            f"{ttft:>10.2f}{e2e:>10.2f}{req.status.value:>10}"
        )


def main() -> None:
    engine = build_engine()
    requests = make_requests()

    print("MiniServe phase 1 demo - token-budget scheduling with chunked prefill")
    print(
        f"config: max_batch_tokens={engine.config.max_batch_tokens} "
        f"max_running_requests={engine.config.max_running_requests} "
        f"num_blocks={engine.config.num_blocks} block_size={engine.config.block_size} "
        f"policy={engine.config.policy}"
    )

    steps = engine.run_to_completion(requests)
    print_timeline(engine, requests, steps)
    print_summary(engine, requests, steps)


if __name__ == "__main__":
    main()
