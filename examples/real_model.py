"""Running a real model through the engine, streaming tokens as they are produced.

The default model is a tiny one with random weights, so the *text* is meaningless but
every code path is real: the scheduler decides what runs, the block manager allocates
physical blocks, and the runner attends over those blocks with a real transformer. Pass
``--model`` for a small instruction model that produces readable text.

Examples::

    python examples/real_model.py
    python examples/real_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --prompt "Explain KV caching"
    python examples/real_model.py --concurrency 3 --max-new-tokens 24
    python examples/real_model.py --blocks 64 --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, Request, WallClock
from mini_infer.engine.engine import StepEventKind
from mini_infer.runner.torch_runner import (
    DEFAULT_MODEL,
    TorchModelRunner,
    engine_config_for_model,
)

DEFAULT_PROMPTS = (
    "The capital of France is",
    "Continuous batching means",
    "A KV cache stores",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="real-model", description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id or path")
    parser.add_argument("--prompt", action="append", help="prompt to generate from (repeatable)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--concurrency", type=int, default=1, help="how many prompts to run at once"
    )
    parser.add_argument("--blocks", type=int, default=128, help="KV pool size in blocks")
    parser.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    parser.add_argument("--max-batch-tokens", type=int, default=64, help="step token budget")
    parser.add_argument("--dtype-bytes", type=int, default=4, choices=(2, 4), help="KV precision")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true", help="print every engine step")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = engine_config_for_model(
        args.model,
        num_blocks=args.blocks,
        block_size=args.block_size,
        max_batch_tokens=args.max_batch_tokens,
        max_running_requests=max(args.concurrency, 1),
        dtype_bytes=args.dtype_bytes,
    )
    runner = TorchModelRunner(config, model=args.model)
    described = runner.describe()
    print(
        f"model  : {described['model']}\n"
        f"shape  : {described['layers']} layers, {described['heads']} heads "
        f"({described['kv_heads']} kv), head_dim {described['head_dim']}, {described['dtype']}\n"
        f"kv pool: {described['blocks']} blocks x {described['block_size']} tokens "
        f"= {described['pool_mib']} MiB\n"
        f"budget : {config.max_batch_tokens} tokens/step, "
        f"{config.max_running_requests} concurrent requests\n"
    )
    if args.temperature > 0:
        print("note: this runner samples greedily; --temperature is accepted but unused\n")

    prompts = args.prompt or list(DEFAULT_PROMPTS[: max(args.concurrency, 1)])
    engine = Engine(config, runner=runner, clock=WallClock())
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            print(f"prompt {index} encoded to nothing; skipping", file=sys.stderr)
            continue
        engine.submit(
            Request(
                prompt_tokens=token_ids,
                max_new_tokens=args.max_new_tokens,
                arrival_time=0.0,
                id=f"p{index}",
            )
        )
        print(f"── prompt {index}: {prompt!r} ({len(token_ids)} tokens)")

    printed: dict[str, str] = {}
    started = time.perf_counter()
    for step in engine.run(max_steps=args.max_steps):
        if args.verbose:
            work = " ".join(
                f"{item.request_id}:{item.kind.value[0]}{item.num_new_tokens}"
                for item in step.output.work
            )
            print(
                f"   step {step.index:>3} {step.duration * 1000:7.2f}ms "
                f"kv={step.kv_utilization:5.1%} run={step.running_requests} "
                f"wait={step.waiting_requests} | {work}"
            )
        for event in step.events:
            if event.kind is not StepEventKind.DECODED:
                continue
            request = engine.requests[event.request_id]
            text = tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
            delta = text[len(printed.get(request.id, "")) :]
            if delta:
                print(f"[{request.id}] {delta}", flush=True)
            printed[request.id] = text

    wall = time.perf_counter() - started
    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    print("\n── results")
    for request in engine.requests.values():
        text = tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
        print(f"[{request.id}] {request.num_generated}/{request.max_new_tokens} tokens: {text!r}")
    print(
        f"\nsteps={metrics.num_steps} generated={metrics.generated_tokens} "
        f"throughput={metrics.output_throughput:.1f} tok/s "
        f"ttft={metrics.mean_ttft * 1e3:.1f}ms tpot={metrics.mean_tpot * 1e3:.1f}ms "
        f"kv_peak={metrics.peak_kv_utilization:.1%} evictions={metrics.num_preemptions} "
        f"wall={wall:.2f}s"
    )
    print(
        "\nThat was one forward pass per step over the whole batch, attention reading the\n"
        "same physical KV blocks the block manager allocated."
    )
    return 0 if metrics.num_finished == metrics.num_requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
