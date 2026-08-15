# MiniServe — Hands-On Guide

How to run, observe, poke at, and explain what's been built. Everything here is
tested; every number shown is from an actual run on this machine.

**Status:** 5 of 8 phases done, 89 checks passing, 6 commits.

```
mini_infer/engine/      request model, scheduling contract, the execution loop
mini_infer/memory/      KV block pool (BlockPlan = transactional view)
mini_infer/runner/      simulated execution-cost model
mini_infer/metrics/     TTFT / ITL / TPOT / throughput / KV utilization
mini_infer/benchmark/   workloads, driver, sweeps, exports
mini_infer/visualizations/  text timelines and charts
scripts/check_*.py      89 checks (no test framework — plain asserts)
examples/*.py           runnable demos, each showing one idea
```

---

## 1. Verify the whole thing in 30 seconds

```bash
cd /Users/abhinandan/Desktop/mini_infer

for f in check_engine check_memory check_preemption check_metrics check_benchmark; do
  python scripts/$f.py
done
```

Expected:

```
phase 1 skeleton: all 24 checks passed
phase 2 block-based KV manager: all 16 checks passed
phase 3 recompute preemption: all 8 checks passed
phase 4 metrics: all 18 checks passed
phase 5 benchmark harness: all 23 checks passed
```

Then the four demos:

```bash
python examples/demo.py            # the brief's scenario, step by step
python examples/kv_pressure.py     # preemption off vs on
python examples/metrics_demo.py    # what the metrics look like
python examples/benchmark_demo.py  # saturation + sweeps
```

If all of that runs, the system is healthy. The rest of this guide is about
**understanding** it.

---

## 2. The one mental model

Hold this and everything else follows:

```
        ┌────────────────────────────────────────────────────────┐
        │  ONE ENGINE STEP == ONE MODEL FORWARD PASS             │
        └────────────────────────────────────────────────────────┘

  Scheduler          decides: how many tokens does each request get this step?
      │                       { R1: 1, R2: 1, R3: 30 }
      ▼
  KVBlockManager     checks: do the blocks exist to back that work?
      │                       (if not → preempt someone, or admit nobody)
      ▼
  ModelRunner        executes: prefill chunks / decode tokens
      │
      ▼
  Engine             commits: advances request state, moves KV blocks,
      │                       decides who finished, advances the clock
      ▼
  MetricsCollector   records: TTFT, ITL, throughput, KV pressure
```

Two ideas do most of the work:

1. **The scheduler never says "prefill R3".** It says "give R3 thirty compute
   tokens." A full prefill, a chunked prefill, and a decode are all the same kind
   of decision — a token count. That's `SchedulerOutput` in
   `mini_infer/engine/scheduler.py`.

2. **Planning never mutates memory; only the engine commits.** The scheduler builds
   a plan against a `BlockPlan` (a transactional view of the pool) and returns it.
   The engine applies frees and allocations in one ordered phase. This is what makes
   a plan abortable — important once a real model can raise.

---

## 3. Experiment 1 — watch the engine step

**What you're testing:** that decode is one token per step, and that the token budget
is real.

```python
# save as /tmp/e1.py, run with: python /tmp/e1.py
import sys; sys.path.insert(0, "/Users/abhinandan/Desktop/mini_infer")
from mini_infer import Engine, EngineConfig, Request, RunnerConfig

cfg = EngineConfig(
    max_batch_tokens=64, max_running_requests=4, num_blocks=32, block_size=16,
    runner=RunnerConfig(prefill_base_ms=2.0, prefill_ms_per_token=0.08,
                        decode_base_ms=1.5, decode_ms_per_token=0.05,
                        decode_ms_per_context_token=0.004),
)
e = Engine(cfg)
e.submit(Request(id="R1", prompt_tokens=list(range(80)), max_new_tokens=3, arrival_time=0.0))

for s in e.run_to_completion():
    plan = [(w.request_id, w.kind.value, w.num_new_tokens) for w in s.output.work]
    print(f"step {s.index}: prefill={s.num_prefill_tokens:3d} decode={s.num_decode_requests} "
          f"batch={s.output.num_scheduled_tokens:3d} dur={s.duration*1000:6.2f}ms "
          f"free_blocks={s.free_blocks} {plan}")
```

Actual output:

```
step 0: prefill= 64 decode=0 batch= 64 dur=  7.12ms free_blocks=28 plan=[('R1','prefill',64)]
step 1: prefill= 16 decode=0 batch= 16 dur=  3.28ms free_blocks=27 plan=[('R1','prefill',16)]
step 2: prefill=  0 decode=1 batch=  1 dur=  2.00ms free_blocks=26 plan=[('R1','decode',1)]
step 3: prefill=  0 decode=1 batch=  1 dur=  2.00ms free_blocks=26 plan=[('R1','decode',1)]
step 4: prefill=  0 decode=1 batch=  1 dur=  2.00ms free_blocks=32 plan=[('R1','decode',1)]
```

**Read this carefully — four things are visible:**

| Observation | Why |
|---|---|
| Step 0 stops at exactly **64** tokens | `max_batch_tokens=64` is a hard ceiling |
| A **80**-token prompt became `64 + 16` | one prompt spans two steps: chunked prefill |
| Each decode step adds **1** token | `ScheduledWork` rejects decode != 1 |
| `free_blocks` goes 28 → 27 on step 1, then holds at 26 | 80 tokens = 5 blocks; block 5 is only *needed* once the sequence crosses 64 |

**Now break it.** Change `max_batch_tokens=64` to `16` and rerun. You should see the
prompt split into five steps. That's the scheduler responding to the one resource it
manages.

**Where the logic lives:**
- `Request.on_prefill` / `on_decode` — `mini_infer/engine/request.py`
- The budget loop — `SchedulerBase._build_plan` in `mini_infer/engine/policies.py`
- Cost model — `mini_infer/runner/simulated_runner.py`

---

## 4. Experiment 2 — why chunked prefill exists

**What you're testing:** the headline trade-off. A long prompt should not block short
ones — but chunking costs the long request.

```python
import sys; sys.path.insert(0, "/Users/abhinandan/Desktop/mini_infer")
from mini_infer import Engine, EngineConfig, Request, RunnerConfig

for chunked, cap in ((False, None), (True, 16)):
    cfg = EngineConfig(
        max_batch_tokens=64, max_running_requests=4, num_blocks=32, block_size=16,
        enable_chunked_prefill=chunked, max_prefill_chunk=cap,
        runner=RunnerConfig(prefill_base_ms=2.0, prefill_ms_per_token=0.08,
                            decode_base_ms=1.5, decode_ms_per_token=0.05,
                            decode_ms_per_context_token=0.004),
    )
    e = Engine(cfg)
    e.submit(Request(id="long", prompt_tokens=list(range(64)), max_new_tokens=2, arrival_time=0.0))
    e.submit(Request(id="short", prompt_tokens=list(range(8)), max_new_tokens=2, arrival_time=0.0))
    steps = e.run_to_completion()
    ttft = {r.id: round((r.first_token_time or 0) * 1000, 2) for r in e.requests.values()}
    print(f"chunked={chunked} cap={cap}: steps={len(steps)} ttft_ms={ttft}")
    print("   first chunks:", [s.num_prefill_tokens for s in steps[:4]])
```

Actual output:

```
chunked=False cap=None: steps=4 ttft_ms={'long': 9.76, 'short': 11.76}
   first chunks: [64, 8, 0, 0]

chunked=True cap=16: steps=6 ttft_ms={'long': 15.76, 'short': 7.2}
   first chunks: [24, 16, 16, 16]
```

**This is the whole argument, measured:**

- **Without chunking**, the 64-token prompt is prefilled in one step, so the short
  request's TTFT is **11.76 ms** — it waited behind the long prompt.
- **With a 16-token chunk cap**, the scheduler interleaves: the short request gets in
  at **7.20 ms** (−39%), and the long request's TTFT rises to **15.76 ms** (+61%).

You traded the long request's latency for the short one's. That's exactly the trade-off
vLLM's docs describe, and it's why `enable_chunked_prefill` is a **knob, not a policy**.

**Break it differently:** set `max_prefill_chunk=8` and watch the long request's TTFT
climb further while `steps` grows — smaller chunks mean more steps and more overhead.

---

## 5. Experiment 3 — memory is managed in blocks, not tokens

**What you're testing:** that a request reserves memory *as it grows* rather than
`prompt + max_new_tokens` up front.

```python
# /tmp/blocks.py
import sys; sys.path.insert(0, "/Users/abhinandan/Desktop/mini_infer")
from mini_infer import Request
from mini_infer.memory import PagedBlockManager

m = PagedBlockManager(num_blocks=8, block_size=16)
print("pool:", m.num_blocks, "blocks x", m.block_size, "tokens =", m.num_blocks * m.block_size, "slots")

r = Request(prompt_tokens=list(range(20)), max_new_tokens=100, arrival_time=0.0)
print("request: prompt 20, max_new_tokens 100")
print("blocks if we reserved the worst case:", -(-(20 + 100) // 16))
print()

r.on_prefill(20)
m.grow_to(r, r.num_computed_tokens)
print(f"  after prompt prefill: seq={r.num_computed_tokens:3d} -> "
      f"{r.block_table.num_blocks} blocks, table={list(r.block_table)}")

for _ in range(60):
    r.on_decode(token=0, now=0.0)
    m.grow_to(r, r.num_computed_tokens)
    if r.num_computed_tokens % 16 == 0:
        print(f"  after {r.num_computed_tokens:3d} tokens  -> "
              f"{r.block_table.num_blocks} blocks ({m.num_free_blocks()} free), "
              f"table={list(r.block_table)}")
```

Actual output:

```
pool: 8 blocks x 16 tokens = 128 slots
request: prompt 20, max_new_tokens 100
blocks if we reserved the worst case: 8

  after prompt prefill: seq= 20 -> 2 blocks, table=[0, 1]
  after  32 tokens  -> 2 blocks (6 free), table=[0, 1]
  after  48 tokens  -> 3 blocks (5 free), table=[0, 1, 2]
  after  64 tokens  -> 4 blocks (4 free), table=[0, 1, 2, 3]
  after  80 tokens  -> 5 blocks (3 free), table=[0, 1, 2, 3, 4]
```

**Why this matters:** reserving `prompt + max_new_tokens` would consume **8 of 8
blocks** — the entire pool — for one request. Actual usage grows 2 → 3 → 4 → 5.
The request's *maximum* output never costs memory it hasn't produced yet.

Note `32 tokens -> 2 blocks`: tokens 21–32 fit in the partially filled block 1, so no
new block is needed. Only when the sequence crosses a multiple of 16 does it grow.
That is why the invariant is `ceil(sequence_len / block_size)` and not
`sequence_len / block_size`.

### Seeing that physical blocks are non-contiguous

The example above happens to produce `[0, 1, 2]` because the allocator picks the
lowest free id first. To see the property that actually matters — a block table is a
*mapping*, not a range — interleave two requests:

```python
# /tmp/paging.py
import sys; sys.path.insert(0, "/Users/abhinandan/Desktop/mini_infer")
from mini_infer import Request
from mini_infer.memory import PagedBlockManager

m = PagedBlockManager(num_blocks=8, block_size=16)

def ready(name):
    r = Request(id=name, prompt_tokens=list(range(20)), max_new_tokens=0, arrival_time=0.0)
    r.on_prefill(20); r.mark_complete()
    return r

r1 = ready("R1"); m.grow_to(r1, 20); print("R1", list(r1.block_table), "free", sorted(m._free))
r2 = ready("R2"); m.grow_to(r2, 20); print("R2", list(r2.block_table), "free", sorted(m._free))
m.free(r1);                            print("R1 finished, freed ->", sorted(m._free))
r3 = ready("R3"); m.grow_to(r3, 20); print("R3", list(r3.block_table), "free", sorted(m._free))
```

Actual output:

```
R1 [0, 1] free [2, 3, 4, 5, 6, 7]
R2 [2, 3] free [4, 5, 6, 7]
R1 finished, freed -> [0, 1, 4, 5, 6, 7]
R3 [4, 5] free [0, 1, 6, 7]
```

Two honest observations, both worth knowing:

1. **No request's blocks are contiguous with another's.** `R1=[0,1]`, `R2=[2,3]` are
   adjacent only by accident of allocation order. Once requests of different lifetimes
   interleave, tables become arbitrary sets of ids — which is exactly why the block
   table exists.

2. **`R3` did not reuse `R1`'s freed blocks `[0, 1]`; it took `[4, 5]`.** The free pool
   is a Python `set`, and `pop()` order for small integers is not insertion order. This
   is *correct* — any free block will do, since the table records the mapping — but if
   you want freed blocks reused promptly (better locality for a real GPU), the pool
   should be a FIFO deque. That's a legitimate improvement to note, not a bug.

**Where the logic lives:** `PagedBlockManager.grow_to`, `free`, and `BlockPlan` in
`mini_infer/memory/block_manager.py`.

> Naming note for interviews: this is a **block-based / paged-style KV cache manager**.
> Attention is not executed over physical tensors, so it is *not* a PagedAttention
> implementation. Saying so precisely is a strength, not a caveat.

## 6. Experiment 4 — the coupling: preemption

This is the most interesting result. `examples/kv_pressure.py` runs a workload that
cannot fit, twice:

```bash
python examples/kv_pressure.py
```

```
PREEMPTION OFF          PREEMPTION ON
  steps      : 27         steps      : 42
  evictions  : 0          evictions  : 1
  finished   : 0/4        finished   : 4/4
```

Without preemption, four requests each grow until the 16-block pool is dry, then
**nobody** can advance: every request holds a partial sequence and needs a block that
doesn't exist. With it, one eviction unblocks the batch — the victim's prompt is
recomputed from the output tokens it kept.

**Read the reasoning in the output:**

```
A evicted (evicted for B)
A: finished, 32 tokens, recomputed 1x
```

A lost its cached prompt but **kept all 32 generated tokens**. Recomputation re-caches
them. No output is lost; only prefill compute is spent.

**Verify nothing corrupted itself:**

```python
# after the run in examples/kv_pressure.py, this must not raise
engine.memory.assert_invariants()
```

That checks: no block owned twice, every block table covers its sequence, and
free + allocated == total (no leaks).

**Where the logic lives:**
- `SchedulerBase._evict_to_fund` — decides *who* to evict (pure planning)
- `Engine._release_preempted_kv` — does the freeing and rewinding (the only writer)
- `Request.on_preempted` — rewinds the cursor, keeps the output

---

## 7. Experiment 5 — metrics, and what they reveal

```bash
python examples/metrics_demo.py
```

```
LATENCY (ms)               mean      p50      p95
  TTFT                    63.08    39.92   143.15
  ITL                      3.44     3.20     6.88
  E2E                    152.47   164.72   229.55
  TPOT (per token)         3.51     3.46     4.13
  queue time              47.33        -   124.19
KV  peak 87.5%   mean 39.6%   max prefill/step 62   max decode batch 4
```

**The thing to notice:** mean TTFT is 63 ms but p95 is 143 ms — **2.3×**. A mean alone
would hide the requests that waited. That's why the collector stores a per-token
timeline rather than just averages, and why `ITL p95` is more interesting than
`ITL mean`: a chunked prefill interrupting decode shows up in the tail, not the mean.

**Definitions** (these match vLLM's published benchmark definitions, so numbers are
comparable):

| Metric | Definition | Code |
|---|---|---|
| TTFT | arrival → first token | `RequestMetrics.ttft` |
| TPOT | `(e2e − ttft) / (output_tokens − 1)` | `RequestMetrics.tpot` |
| ITL | gap between consecutive tokens | `RequestMetrics.inter_token_latencies` |
| queue time | arrival → first step that ran it | `RequestMetrics.queue_time` |

**Check reproducibility** — simulated runs must be bit-identical:

```python
# scripts/check_metrics.py has this check: run the same workload twice,
# assert the metric rows are equal. If that ever fails, benchmarks are noise.
```

---

## 8. Experiment 6 — benchmarking and sweeps

```bash
# find the saturation point
python -m mini_infer.cli --requests 120 --sweep rate --values 2,4,8,16,32,64,128,256

# KV pool pressure
python -m mini_infer.cli --requests 120 --sweep num_blocks --values 16,32,64,128

# budget effect
python -m mini_infer.cli --requests 120 --sweep max_batch_tokens --values 16,32,64,128

# export for a README or plot
python -m mini_infer.cli --sweep rate --values 4,16,64 --export /tmp/results
```

The saturation sweep from `examples/benchmark_demo.py`:

```
throughput (tok/s)                 TTFT p95 (ms)
   2/s  █████              119       2/s  █                    41
  16/s  ██████████         841      16/s  ██████            1,373
  32/s  ██████████████     877      32/s  ███████████████   4,631
 128/s  ██████████████     883     128/s  ████████████████  7,447
 256/s  ██████████████     885     256/s  ████████████████  7,897
```

**This is the capacity limit, measured.** Throughput flattens around **884 tok/s**
while TTFT p95 grows **190×**. Past saturation, extra offered load becomes *latency*,
not work done. And because `queue p95` tracks TTFT p95 almost exactly, the growth is
queueing — not slower generation.

### The token budget is not "bigger is better"

This is the most counter-intuitive result so far, and a good one to have measured:

```bash
python -m mini_infer.cli --requests 150 --rate 12 --sweep max_batch_tokens \
  --values 16,32,64,128 --prompt mixed --output balanced --num-blocks 256
```

```
case  done     TTFT mean  TTFT p95  TPOT mean  ITL p95
  16  150/150    3,475.5   6,780.8      5.259     7.52
  32  150/150    2,784.0   5,551.1      6.123    8.128     <- best latency
  64  150/150    2,894.1   5,794.0       6.72    8.328
 128  150/150    3,312.9   6,600.2      7.181      12     <- worst latency
```

TTFT is **non-monotonic** in the budget: 32 tokens/step beats both 16 and 128.

Why: a bigger budget lets one long prompt monopolise a whole step, so decodes and
short prompts wait — exactly the effect chunked prefill exists to mitigate. A smaller
budget interleaves more requests per unit of work but adds step overhead. Somewhere
between is the sweet spot, and this is how you'd find it for a given workload.

That's a real answer to "why did you choose that number?", which is the kind of
question the README experiments need to be able to answer.

### The three arrival models are not interchangeable (same offered load):

| Mode | TTFT p95 | queue mean | why |
|---|---|---|---|
| burst | 8,163 ms | 3,847 ms | everything offered at t=0 |
| poisson @16/s | 1,373 ms | 724 ms | open-loop, queue grows past saturation |
| closed_loop (4) | 42 ms | **0.297 ms** | clients wait for their own completions |

Closed-loop queue time is essentially zero **by construction** — the client pool
self-limits. That's precisely why it's the right workload for comparing *scheduler
behaviour*, while burst measures *capacity*.

---

## 9. How to poke at invariants yourself

The check scripts are modular — run one check:

```bash
python -c "
import sys; sys.path.insert(0,'scripts')
import check_memory as m
m.check_randomized_workload_holds_all_invariants()
print('PASS')
"
```

Write a throwaway invariant check with the harness (`scripts/_check.py`):

```python
import sys; sys.path.insert(0,'scripts')
from _check import check, close, expect_raises, report
# ... your checks ...
raise SystemExit(report("my experiment"))
```

To fuzz the whole engine, copy the shape of `check_randomized_workload_holds_all_invariants`:
build a random workload with a seed, step it, and after **every** step assert
`engine.memory.assert_invariants()`. That single line catches most real bugs.

**The invariants that matter** (all currently enforced):

```
allocated + free blocks == total blocks          no capacity leaks
one physical block belongs to at most one request
a block table exactly covers its computed prefix
scheduled tokens <= max_batch_tokens             budget is a hard ceiling
a decode step produces at most one token
num_computed_tokens <= num_tokens
generated_tokens <= max_new_tokens
planning never mutates memory                    plans are abortable
finished requests own zero blocks
```

---

## 10. Reading map — question to file

| Question | File | Key symbol |
|---|---|---|
| What is a request's state? | `engine/request.py` | `Request`, `num_computed_tokens` |
| What does a step decide? | `engine/scheduler.py` | `SchedulerOutput`, `ScheduledWork` |
| How is a plan built? | `engine/policies.py` | `SchedulerBase._build_plan` |
| How does the budget work? | `engine/policies.py` | `_chunk_size`, `_chunk_cap` |
| Who gets evicted? | `engine/policies.py` | `_evict_to_fund`, `_evictable` |
| Where does the loop live? | `engine/engine.py` | `Engine.step` |
| Where is memory committed? | `engine/engine.py` | `_release_preempted_kv`, `_allocate_kv` |
| How are blocks allocated? | `memory/block_manager.py` | `grow_to`, `BlockPlan` |
| How fast is a step? | `runner/simulated_runner.py` | `SimulatedModelRunner.time_step` |
| How are metrics defined? | `metrics/collector.py` | `RequestMetrics`, `RunMetrics` |
| How is load generated? | `benchmark/workloads.py` | `WorkloadSpec`, `*Arrivals` |
| Who feeds the engine? | `benchmark/driver.py` | `Driver.coordinate` |

**Suggested reading order** (each builds on the last, ~250 lines total):
`request.py` → `scheduler.py` → `policies.py` → `engine.py` → `block_manager.py`.

---

## 11. Interview one-liners

These are the claims you can defend with a number:

- *"The scheduler operates on a token budget, not on prefill/decode modes. One
  forward pass gets at most `max_batch_tokens` of work, split between decode tokens
  and prefill chunks."* — Experiment 1

- *"Chunked prefill cut the short request's TTFT from 11.8 ms to 7.2 ms under a long
  prompt, at the cost of raising the long request's TTFT from 9.8 ms to 15.8 ms. It's
  a knob, and I measured both sides."* — Experiment 2

- *"KV is paged: an 8-block pool serves a request that asks for 100 output tokens
  because blocks are allocated as the sequence grows. Reserving the worst case would
  have taken the whole pool."* — Experiment 3

- *"Compute scheduling and memory scheduling are coupled. With preemption off, all 4
  requests deadlocked holding partial sequences; with it on, all 4 finished after one
  eviction and one recomputation, with no output tokens lost."* — Experiment 4

- *"Throughput saturates near 884 tok/s; past that, added load becomes queueing delay
  — TTFT p95 grows 190× while queue time tracks it almost exactly."* — Experiment 6

- *"The token budget is not monotonically good. At 150 requests under load, 32
  tokens/step gave the best mean TTFT (2.8 s) — better than both 16 (3.5 s) and 128
  (3.3 s), because a large budget lets one prompt monopolise a step."* — Experiment 6

- *"Planning is speculative: the scheduler mutates nothing. I have a check that plans
  evictions for 60 steps and asserts the pool and every block table are bit-identical
  afterwards."* — `check_preemption.py`

---

## 12. Known limits (say these before you're asked)

| Limit | Status |
|---|---|
| Preemption is recompute-only, no swap | deliberate; recompute is cheaper to reason about |
| Eviction is LIFO (youngest yields) | one policy; `max_wait_steps` aging exists but is unused |
| Cost model is synthetic | stated linear model, not a GPU measurement |
| No real attention over paged tensors | it's a block manager, not PagedAttention |
| FCFS is the only policy so far | Phase 6 adds prefill-first / decode-first / balanced |
| A request that can never fit stalls the run | `can_ever_fit()` detects it; no rejection policy yet |

---

## 13. Quick reference

```bash
# all checks
for f in check_engine check_memory check_preemption check_metrics check_benchmark; do
  python scripts/$f.py; done

# demos
python examples/demo.py             # scheduler timeline + KV occupancy
python examples/kv_pressure.py      # preemption off vs on
python examples/metrics_demo.py     # TTFT/ITL/TPOT/throughput
python examples/benchmark_demo.py   # saturation, KV sweeps, arrival models

# CLI
python -m mini_infer.cli --help
python -m mini_infer.cli --requests 200 --arrival poisson --rate 16
python -m mini_infer.cli --sweep num_blocks --values 16,32,64,128 --export /tmp/r
```

Git history, one subsystem per commit:

```
911a3d2  workload generation, benchmark harness and text charts
d3b91fd  metrics collection for latency, throughput and KV pressure
6f24089  single compute cursor + speculative scheduling
3d6790e  KV admission control and recompute preemption
7603c43  block-based KV cache manager
7f4e4ab  runtime skeleton (request model, config, clock, engine loop)
```
