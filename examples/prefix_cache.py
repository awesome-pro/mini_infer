"""Prefix caching: KV computed once, reused by every request with the same prefix.

Real traffic shares prefixes constantly — a system prompt, a tool preamble, a chat
history — and a block-based cache can hand the same physical blocks to every request
that starts with those tokens, so the prefix is never computed twice.

Run with::

    python examples/prefix_cache.py
    python examples/prefix_cache.py --requests 120 --rate 12 --prefix 960
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_infer import Engine, EngineConfig
from mini_infer.benchmark.driver import Driver
from mini_infer.benchmark.workloads import Constant, LengthProfile, WorkloadSpec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prefix-cache", description=__doc__)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--rate", type=float, default=8.0, help="arrival rate, requests/s")
    parser.add_argument("--prefix", type=int, default=480, help="shared prefix tokens")
    parser.add_argument("--tail", type=int, default=8, help="unique tokens per request")
    parser.add_argument("--output", type=int, default=24, help="tokens to generate")
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=64, help="tokens per step")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args(argv)


def shared_workload(args: argparse.Namespace):
    """One workload, built once, with every prompt rewritten to share a prefix.

    Built once on purpose: rebuilding it would resample both the lengths and the arrival
    times, and the two runs would no longer be comparable.
    """
    spec = WorkloadSpec(
        num_requests=args.requests,
        prompt=LengthProfile(Constant(args.prefix + args.tail)),
        output=LengthProfile(Constant(args.output)),
        arrival="poisson",
        rate=args.rate,
        seed=0,
    )
    workload = spec.build()
    prefix = list(range(1, args.prefix + 1))
    for index, request in enumerate(workload.requests):
        # A unique tail per request, so the cache hit stops at the shared prefix.
        tail = [10_000 + index * args.tail + offset for offset in range(args.tail)]
        request.prompt_tokens = prefix + tail
    return workload, prefix


def run(prefix_cache: bool, args: argparse.Namespace) -> dict:
    workload, _ = shared_workload(args)
    config = EngineConfig(
        max_batch_tokens=args.budget,
        max_running_requests=args.concurrency,
        num_blocks=args.blocks,
        block_size=args.block_size,
        enable_prefix_cache=prefix_cache,
    )
    engine = Engine(config)
    result = Driver(engine, workload).run()
    metrics = result.metrics
    memory = engine.memory
    assert memory is not None
    return {
        "label": "on" if prefix_cache else "off",
        "prefill_tokens": sum(step.num_prefill_tokens for step in engine.history),
        "reused": metrics.cached_prefix_tokens,
        "hits": metrics.prefix_cache_hits,
        "ttft": metrics.mean_ttft * 1e3,
        "p95_ttft": metrics.p95_ttft * 1e3,
        "e2e": metrics.mean_e2e * 1e3,
        "throughput": metrics.output_throughput,
        "evictions": metrics.num_preemptions,
        "steps": metrics.num_steps,
        "shared": memory.num_shared_blocks,
        "cached": memory.num_cached_blocks,
        "free": memory.num_free_blocks(),
        "finished": metrics.num_finished,
        "stalled": result.stalled,
    }


def report(rows: list[dict], args: argparse.Namespace) -> None:
    print(
        f"\n{args.requests} requests, Poisson {args.rate:g}/s, "
        f"{args.prefix}-token shared prefix + {args.tail} unique tokens,\n"
        f"{args.blocks} blocks x {args.block_size} tokens, budget {args.budget}, "
        f"{args.concurrency} concurrent\n"
    )
    header = (
        f"{'cache':>5}  {'prefill tok':>11}  {'reused':>8}  {'hits':>5}  "
        f"{'TTFT mean':>10}  {'TTFT p95':>9}  {'E2E mean':>9}  {'out tok/s':>10}  {'steps':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:>5}  {row['prefill_tokens']:>11,}  {row['reused']:>8,}  "
            f"{row['hits']:>5,}  {row['ttft']:>9,.1f}ms  {row['p95_ttft']:>8,.1f}ms  "
            f"{row['e2e']:>8,.1f}ms  {row['throughput']:>10,.1f}  {row['steps']:>6,}"
        )

    off, on = rows
    saved = off["prefill_tokens"] - on["prefill_tokens"]
    share = saved / off["prefill_tokens"] if off["prefill_tokens"] else 0.0
    print(
        f"\nThe cache skipped {saved:,} of {off['prefill_tokens']:,} prefill tokens "
        f"({share:.0%}), and mean E2E latency went from {off['e2e']:,.1f}ms to "
        f"{on['e2e']:,.1f}ms."
    )
    print(
        f"{on['hits']} requests inherited a prefix, and {on['cached']} blocks stayed cached\n"
        f"after every request finished -- still counted as free capacity "
        f"({on['free']}/{args.blocks})."
    )
    print(
        "\nTwo things worth noticing:\n"
        "  * A request can only reuse KV that already exists when it is scheduled, so\n"
        "    requests arriving together cannot share: the first one computes the prefix.\n"
        "    Staggered arrivals are what make a shared prefix pay, which is why the\n"
        "    saving here grows with the arrival rate rather than with the request count.\n"
        "  * A hit is planned like an allocation rather than bolted on afterwards: the\n"
        "    request's remaining work and block requirement both shrink, and admission\n"
        "    control still decides whether it runs. Otherwise a request that a cached\n"
        "    prefix makes cheap to evict would be evicted, re-admitted and evicted again."
    )


def show_concurrent_sharing(args: argparse.Namespace) -> None:
    """Two live requests reading the same physical KV, which is the whole idea."""
    from mini_infer import Request

    _, prefix = shared_workload(args)
    config = EngineConfig(
        max_batch_tokens=args.budget,
        max_running_requests=2,
        num_blocks=args.blocks,
        block_size=args.block_size,
        enable_prefix_cache=True,
    )
    engine = Engine(config)
    first = engine.submit(
        Request(prompt_tokens=[*prefix, 1, 1, 1, 1], max_new_tokens=8, arrival_time=0.0, id="first")
    )
    # Let the first request finish its prompt: the budget only allows a chunk per step,
    # and a cached prefix has to exist before anyone can inherit it.
    while not first.is_prefill_complete:
        engine.step()

    second = engine.submit(
        Request(
            prompt_tokens=[*prefix, 2, 2, 2, 2],
            max_new_tokens=8,
            arrival_time=0.0,
            id="second",
        )
    )
    engine.step()  # second inherits those blocks instead of computing them
    assert second.num_computed_tokens >= len(prefix), "the hit should cover the whole prefix"

    memory = engine.memory
    assert memory is not None
    shared = [
        block
        for block in second.block_table.physical_blocks
        if first.id in memory.owners_of(block)
    ]
    print(
        f"\nTwo requests, one prefix, both alive:\n"
        f"  {second.id} holds {len(shared)} blocks that {first.id} is holding too, e.g.\n"
        f"  block {shared[0] if shared else '<none>'} is in both block tables, and both\n"
        f"  requests read\n"
        f"  their keys and values from that one physical place. No copy was made."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [run(False, args), run(True, args)]
    report(rows, args)
    show_concurrent_sharing(args)
    return 0 if all(row["finished"] == args.requests and not row["stalled"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
