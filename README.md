# MiniServe: a miniature LLM inference runtime

**Continuous batching, token-budget scheduling, chunked prefill and block-based KV
management, built from scratch to study what a serving scheduler actually trades away.**

```text
Python 3.12+   ·   zero runtime dependencies   ·   155 checks, no test framework
5 scheduling policies   ·   3 arrival processes   ·   14 figures, all generated from real runs
```

The engine turns a stream of concurrent generation requests into a sequence of model
forward passes. Every step it decides how many compute tokens each request gets and which
physical KV blocks back them. It is a scheduler and memory-management project: model
execution is a stated cost model, and the decisions, the memory and the measurements are
where the behaviour shows up.

---

## Headline result

120 requests, Poisson arrivals at 16/s, prompts 4–512 tokens, outputs 4–256, a 64-token
step budget, 4 concurrent requests, a 256-block KV pool:

```text
Requests:                  120 / 120 finished
Prompt tokens:            11,252
Generated tokens:          7,159
Engine steps:              2,046

Throughput:                845 tok/s
Mean TTFT:                 575 ms          (p50 515 ms, p95 1,200 ms)
Mean TPOT:                 4.02 ms         (ITL p95 6.56 ms)
Mean E2E:                  823 ms          (p95 1,501 ms)
Queue time (mean):         559 ms          (97% of mean TTFT)
KV peak utilization:       29.7%           (mean 13.9%, 5.4% internal fragmentation)
Decode batch (mean/max):   3.5 / 4
```

Queue time is 97% of TTFT. In a saturated serving system most of the latency is admission
delay, not generation, which is why these experiments move the scheduler rather than the
kernel.

> **What these numbers are.** Modelled time from a stated linear cost function
> (`prefill = 2.0 ms + 0.08·tokens`, `decode = 1.5 ms + 0.05·requests + 0.004·context`),
> not measurements of a GPU. The engine, scheduler, KV manager and metrics are real; the
> execution cost is a model, so the results are reproducible and comparative rather than
> absolute. Everything below comes out of the scripts in this repository, see
> [Reproduce](#reproduce).

---

## Architecture

```text
   Requests ──▶┌─────────────────┐
               │  Request queue  │   burst · Poisson · closed-loop clients
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │    Scheduler    │   one decision per step, per request:
               │                 │   "give r3 thirty compute tokens"
               │  token budget   │
               │  continuous     │   SchedulerOutput{ r1:1, r2:1, r3:30 }
               │  chunked prefill│   + evictions + block allocations
               │  preemption     │
               └────────┬────────┘
                        │  the plan is speculative:
          ┌─────────────┴──────────────┐
          ▼                            ▼
  ┌───────────────┐            ┌───────────────┐
  │ KV block      │            │ Model runner  │
  │ manager       │            │               │
  │ block tables  │            │ prefill chunk │
  │ alloc / free  │            │ decode token  │
  │ PagePlan      │            └───────┬───────┘
  └───────┬───────┘                    │
          └─────────────┬──────────────┘
                        ▼
               ┌─────────────────┐
               │     Engine      │   commits state, moves blocks, decides
               └────────┬────────┘   who finished, advances the clock
                        ▼
               ┌─────────────────┐
               │     Metrics     │   TTFT · ITL · TPOT · E2E · queue · throughput · KV
               └─────────────────┘
```

Four rules hold the design together:

1. **One engine step is one model forward pass.** Everything the engine does is work
   inside a step.
2. **The scheduler speaks token grants, not modes.** It never says "prefill r3"; it says
   "give r3 thirty tokens". Full prefill, chunked prefill and decode are then the same
   kind of decision, expressed by the same contract.
3. **Planning is speculative.** The scheduler decides against a transactional `BlockPlan`
   and mutates nothing. A plan that is abandoned cannot leave physical memory moved.
4. **Memory is physical, not a counter.** KV lives in fixed-size blocks with per-request
   block tables, allocated as a sequence grows and freed when it ends.

---

## Quickstart

```bash
git clone <this repo> && cd mini_infer
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # no runtime dependencies

# 155 checks, no test framework: nine scripts of plain asserts
for f in check_engine check_memory check_preemption check_metrics check_benchmark \
         check_policies check_prefix_cache check_visualizations check_torch_runner; do
  python scripts/$f.py; done

# a real model through the scheduler, streaming tokens (needs the torch extra)
pip install -e ".[torch]"
python examples/real_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct

# runnable demos of the simulated runtime, each showing one idea
python examples/demo.py                # a scheduler timeline and KV occupancy
python examples/kv_pressure.py         # preemption off vs on
python examples/policy_comparison.py   # the five policies, head to head
python examples/benchmark_demo.py      # saturation, KV sweeps, arrival models
python examples/prefix_cache.py        # the same prefix computed once, not 120 times

# benchmark anything from the CLI
python -m miniserve.cli --requests 200 --arrival poisson --rate 16
python -m miniserve.cli --sweep policy --values fcfs,prefill_first,balanced,static
python -m miniserve.cli --sweep max_batch_tokens --values 16,32,64,128 --export /tmp/r
python -m miniserve.cli --prefix-cache --requests 120 --rate 12
python -m miniserve.cli --sweep prefix_cache --values off,on --requests 120 --rate 12

# regenerate every figure in this README
pip install -e ".[plot]" && python scripts/make_figures.py
```

`TESTING.md` is the hands-on guide: nine experiments with expected output, a reading
order, and the invariants to poke at yourself.

---

## What a run looks like

Four requests, a 32-token step budget, 16-token prefill chunks, a 24-block pool. A `P<n>`
cell is a prefill chunk of *n* tokens for that request; each `D1` is one output token:

![timeline](figures/timeline.png)

Prefill chunks interleave with decode (steps 1–5 admit new requests while others
generate), the 32-token budget is never exceeded, requests finish at different steps and
their slots go idle, and the tail runs with only one or two requests active.

The same run's memory and load. The dashed line is internal fragmentation, reserved block
capacity holding no tokens, which comes and goes as sequences start, cross block
boundaries and finish:

![KV and load](figures/kv_and_load.png)

And the latency spread a single run produces, from the 300-request run at Experiment 1's
16 req/s point:

![latency distributions](figures/latency_distributions.png)

The TTFT histogram is bimodal: a fast cluster that arrived when the engine was idle, and a
queueing mode around 2.2 s. ITL stays in a tight 3–7 ms band. Latency under load is mostly
a story about when you arrived.

---

## Experiments

Every figure is produced by `scripts/make_figures.py`, from a fixed seed, on every run.

### 1. Capacity: throughput saturates, latency does not

![saturation](figures/saturation.png)

| Offered load | Throughput | Mean TTFT | p95 TTFT | p95 queue |
|---|---|---|---|---|
| 2 req/s | 122.9 tok/s | 13.8 ms | 30.7 ms | 0 ms |
| 4 req/s | 245.4 tok/s | 15.9 ms | 35.4 ms | 0 ms |
| 8 req/s | 489.5 tok/s | 36.3 ms | 126.8 ms | 101.5 ms |
| 16 req/s | 845.1 tok/s | 575.2 ms | 1,200.2 ms | 1,186.8 ms |
| 32 req/s | 862.2 tok/s | 2,176.9 ms | 4,157.3 ms | 4,142.5 ms |
| 64 req/s | 863.3 tok/s | 3,053.7 ms | 5,861.1 ms | 5,846.4 ms |
| 256 req/s | 864.0 tok/s | 3,707.3 ms | 7,141.4 ms | 7,125.5 ms |

Throughput flattens near 864 tok/s at ~32 req/s while p95 TTFT grows 233× (30.7 ms to
7,141 ms). Past saturation, extra offered load becomes waiting. The queue-time column
tracking TTFT almost exactly is the evidence.

### 2. Static vs continuous batching

![static vs continuous](figures/static_vs_continuous.png)

| Offered load | Continuous | Static | | p95 TTFT continuous | p95 TTFT static |
|---|---|---|---|---|---|
| 2 req/s | 122.9 tok/s | 122.9 tok/s | | 30.7 ms | 349.3 ms |
| 4 req/s | 245.4 tok/s | 245.4 tok/s | | 35.4 ms | 622.1 ms |
| 8 req/s | 489.5 tok/s | 474.8 tok/s | | 126.8 ms | 1,362.5 ms |
| 16 req/s | 845.1 tok/s | 492.4 tok/s | | 1,200.2 ms | 6,667.6 ms |
| 32 req/s | 862.2 tok/s | 492.3 tok/s | | 4,157.3 ms | 10,105.0 ms |

Static batching, meaning form a batch and run it to completion before admitting the next,
caps at 492 tok/s and is 5.6× worse on p95 TTFT at 16 req/s. A slot that frees stays idle
until the whole batch drains. Below ~8 req/s the two are identical, because the engine is
not saturated and there is nothing to refill.

With homogeneous lengths the two policies produce identical numbers, since the whole batch
finishes on one step and no slot is ever idle. Static batching loses when the batch is
heterogeneous, which the mixed workload above is. `check_policies.py` asserts both the
difference and the identity.

### 3. Chunked prefill

![chunking](figures/chunking.png)

| Prefill policy | Mean TTFT | p95 ITL | Throughput | Largest prefill step |
|---|---|---|---|---|
| No cap (budget-sized chunks) | 729.0 ms | 6.56 ms | 872.9 tok/s | 64 tokens |
| Cap 64 | 729.0 ms | 6.56 ms | 872.9 tok/s | 64 tokens |
| Cap 32 | 660.4 ms | 5.81 ms | 883.9 tok/s | 62 tokens |
| Cap 16 | **640.0 ms** | **5.72 ms** | **888.4 tok/s** | 48 tokens |

Chunking at 16 tokens improves every aggregate: mean TTFT −12%, p95 ITL −13%, throughput
+2%. The textbook trade-off predicts the opposite, and the reason is this workload. With
prompts up to 512 tokens and a 64-token budget, letting one request consume a whole step
holds up admission for everyone behind it, so shorter chunks admit more requests per unit
time.

Per request the trade is still there and points the other way for long prompts: the short
request's TTFT improves while the long request's worsens, measured in `TESTING.md`
Experiment 2. Aggregate curves hide that.

### 4. The token budget is not "bigger is better"

![budget](figures/budget.png)

| `max_batch_tokens` | Mean TTFT | p95 TTFT | Mean TPOT | p95 ITL | Throughput |
|---|---|---|---|---|---|
| 16 | 3,475.5 ms | 6,780.8 ms | 5.26 ms | 7.52 ms | 505.9 tok/s |
| **32** | **2,784.0 ms** | **5,551.1 ms** | 6.12 ms | 8.13 ms | **541.9 tok/s** |
| 64 (default) | 2,894.1 ms | 5,794.0 ms | 6.72 ms | 8.33 ms | 531.5 tok/s |
| 128 | 3,312.9 ms | 6,600.2 ms | 7.18 ms | 12.00 ms | 507.1 tok/s |

TTFT is non-monotonic in the step budget: 32 tokens/step beats both 16 and 128, and it is
also the throughput peak. A larger budget lets one long prompt monopolise a step and
pushes decodes and short prompts out; a smaller one adds per-step overhead. The configured
default of 64 is not the best value for this workload.

### 5. KV pressure

![KV pressure](figures/kv_pressure.png)

120 requests, 16 concurrency, 48-token prompts, 64-token outputs (~7 blocks each):

| KV pool | Throughput | p95 TTFT | Evictions | Finished |
|---|---|---|---|---|
| 8 blocks | 461.5 tok/s | 8,714.1 ms | 822 | 120/120 |
| 16 blocks | 831.5 tok/s | 1,744.2 ms | 554 | 120/120 |
| 32 blocks | 1,033.7 tok/s | 143.9 ms | 68 | 120/120 |
| 64 blocks | 1,036.0 tok/s | 11.7 ms | 1 | 120/120 |
| 128 blocks | 1,036.0 tok/s | 11.7 ms | 0 | 120/120 |

Below the working set, the pool sets the throughput ceiling. An 8-block pool needs 822
recomputations to finish the work a 64-block pool finishes with 1, at 8.7 seconds of p95
TTFT against 12 ms.

Completion stays at 120/120 in every configuration. Recompute preemption converts memory
pressure into work and latency instead of failed requests: the victim's generated tokens
are kept and only its prompt is recomputed.

### 6. Block size

![block size](figures/block_size.png)

| Tokens per block | Mean fragmentation | Max fragmentation | p95 TTFT | Throughput |
|---|---|---|---|---|
| 8 | 2.6% | 8.9% | 1,200 ms | 845 tok/s |
| 16 (default) | 5.4% | 20.4% | 1,200 ms | 845 tok/s |
| 32 | 10.2% | 29.8% | 1,200 ms | 845 tok/s |
| 64 | 18.6% | 42.2% | 1,200 ms | 845 tok/s |

Internal fragmentation scales with block size, 2.6% at 8 tokens to 18.6% at 64. Latency
and throughput do not move, because the cost model charges nothing per block. So in this
runtime block size is a memory-efficiency knob and nothing else. A real implementation
would price the block-table and gather overhead that makes very small blocks unattractive.

The figure also shows two ways to sweep block size. Holding the pool at 128 blocks varies
capacity as well (1024 to 8192 tokens); holding capacity at 2048 tokens does not. At 8
tokens/block the difference is visible, the pool sweep gets 826 tok/s against 845, and at
16+ both are above the working set and identical. A block-size sweep that fixes the block
count is also a capacity sweep.

### 7. Policy comparison

![policies](figures/policies.png)

One saturating workload (200 requests, Poisson 64/s), four policies:

| Policy | Throughput | Mean TTFT | p95 TTFT | p95 ITL |
|---|---|---|---|---|
| `fcfs` (= `decode_first`) | 1,150.7 tok/s | 3,549.1 ms | 7,121.8 ms | 8.24 ms |
| `prefill_first` | 1,039.0 tok/s | 4,148.9 ms | 8,231.7 ms | 12.55 ms |
| `balanced` | 1,149.5 tok/s | 3,555.5 ms | 7,138.9 ms | 8.25 ms |
| `static` | 500.8 tok/s | 10,859.4 ms | 21,051.4 ms | 3.20 ms |

Under a saturated queue, `prefill_first` is worse on every latency metric. It loses
throughput (1,039 against 1,151 tok/s), and since TTFT here is dominated by queue depth,
that loss costs more than prefill priority wins back. `balanced` matches `fcfs` because
with only 4–8 decoders its 16-token prefill floor is never binding.

Experiment 8 isolates the mechanism instead of averaging it away.

### 8. Decode priority is a trade, not a win

![policy trade-off](figures/policy_tradeoff.png)

8 requests are decoding and already fill the step when a 64-token prompt arrives:

| Policy | Late prompt's TTFT | Delay to the existing decoders |
|---|---|---|
| `decode_first` | 49.4 ms | 2.6 ms |
| `prefill_first` | **20.4 ms** | **17.4 ms** |
| `balanced` | 49.3 ms | 2.6 ms |
| `static` | not served within 400 steps | 2.0 ms |

Prefill priority starts the new request 2.4× sooner and makes the requests already in
flight wait 6.7× longer for their next token. The two policies choose different victims.
`static` never serves the late request within the window because no slot frees until its
batch drains.

### 9. Fairness: bounding the worst wait

![fairness](figures/fairness.png)

40 requests arrive at once with an 8-token budget and 64 slots, so prefills compete with
a saturated decode batch:

| Configuration | Worst wait for a first token |
|---|---|
| `decode_first` | 212 steps |
| `balanced`, reservation only (2 tokens/step) | 141 steps |
| `balanced`, ageing only (`max_wait_steps=8`) | 40 steps |
| `balanced`, both | 40 steps |

Each knob has one job: `prefill_reservation` is a floor the decodes may not spend, and
`max_wait_steps` decides which waiter the floor is spent on. Together they cut the worst
wait 5× (212 to 40 steps) with essentially unchanged throughput, 2,420 tok/s for
`decode_first` against 2,399–2,485 for the balanced variants.

Decode-first does not starve prefills into never running. A prefill can be granted zero
tokens for a step, repeatedly, which is why the worst wait reaches 212 steps, but every
request is eventually served: admission is arrival-ordered and every output budget is
finite, so the decode batch always drains. The failure mode is latency, not deadlock, and
ageing is what bounds it.

### 10. The arrival process changes what "the same load" measures

![arrival models](figures/arrival_models.png)

| Arrival model | Throughput | p95 TTFT | Mean queue | p95 ITL |
|---|---|---|---|---|
| Burst (all at t=0) | 879.5 tok/s | 7,687.2 ms | 3,847.8 ms | 6.88 ms |
| Poisson @16/s | 845.1 tok/s | 1,200.2 ms | 559.3 ms | 6.56 ms |
| Closed loop (4 clients) | 639.4 tok/s | 34.9 ms | **0.059 ms** | 4.20 ms |

Closed-loop queue time is essentially zero by construction: a client only issues its next
request after the previous one finishes, so the offered load self-limits. That makes it
the right workload for comparing scheduler behaviour, while burst measures capacity and
Poisson models an open-loop service. Reporting one number without naming the arrival model
is not reporting a result.

### 11. Prefix caching: KV computed once, reused by everyone

![prefix caching](figures/prefix_cache.png)

120 requests, each a 480-token shared prefix followed by its own 8-token tail, Poisson
arrivals, 256 blocks, a 64-token budget:

| Offered load | Prefill tokens (off → on) | Mean TTFT (off → on) | Mean E2E (off → on) | Throughput (off → on) |
|---|---|---|---|---|
| 2 req/s | 58,560 → **1,440** | 64.3 → **7.3 ms** | 199.6 → 134.9 ms | 65.8 → 65.9 tok/s |
| 4 req/s | 58,560 → **1,472** | 68.4 → **8.3 ms** | 227.9 → 150.8 ms | 131.2 → 131.5 tok/s |
| 8 req/s | 58,560 → **1,472** | 110.0 → **34.5 ms** | 320.5 → 230.1 ms | 261.0 → 261.9 tok/s |
| 16 req/s | 58,560 → **1,472** | 1,319.6 → 1,104.1 ms | 1,572.4 → 1,398.5 ms | 389.1 → **403.3 tok/s** |
| 32 req/s | 58,560 → **1,472** | 3,084.8 → 2,859.6 ms | 3,339.1 → 3,156.2 ms | 389.4 → **404.4 tok/s** |

The prefix is computed once instead of 120 times, which removes 97% of all prefill work,
and 119 of 120 requests inherit KV someone else already computed. Mean TTFT falls 89% at
2 req/s, and throughput rises slightly because the freed compute goes to decoding.

The benefit is largest where the engine is not yet saturated and shrinks as queueing takes
over. At 32 req/s the queue is what the request waits for, so a cache, being a compute
optimisation, has little left to give.

Two design points make it safe.

**Only full blocks are cached, and a hit lands on a block boundary.** A cached block is
immutable and the block a request writes next is always one it owns outright, so there is
no copy-on-write and no shared block is ever written. Contents are hashed together with
the hash of the block before them, so a block only matches at the same absolute positions.
Rotary positions are baked into the KV, so matching on content alone would be wrong.

**A hit is planned, not bolted on.** The scheduler shrinks the request's remaining work
and block requirement and asks the pool whether the rest of the sequence fits before
sharing anything. The engine commits the attachment before it allocates. Planning it this
way is what stops the failure mode the first implementation had: a request that a cached
prefix makes cheap to evict gets evicted, re-admitted and evicted again, forever. Under
real pressure a request waits its turn and takes the prefix later.

Two limits, both measured above. Requests that arrive together cannot share, because
nothing has been computed yet when the second one is scheduled, so staggered arrivals are
what make a shared prefix pay. And cached blocks still occupy the pool, they are merely
evictable, so the cache competes with live requests for capacity.

---

## How the scheduler works

**One step, one budget.** Every step plans at most `max_batch_tokens` of work, split
between decode tokens (one per decoding request) and prefill chunks. The budget is a hard
ceiling and the chunk cap can only lower a chunk. A prompt longer than the budget is split
rather than stalled, because refusing to split would deadlock.

**Two passes, speculative.** The scheduler builds a plan, and if it does not fit the KV
pool it names a victim, marks the eviction in the plan only, and rebuilds. The engine then
commits: free the victims' blocks, rewind them, release anyone who finished, allocate for
the work that ran, in that order. Nothing physical moves while planning.

**Preemption by recomputation.** A victim loses its cached prompt, its cursor rewinds to
zero and its block table is cleared, but it keeps every token it generated. It is
re-admitted later and recomputes its prompt. Victims are chosen youngest-first (LIFO), and
only an older request may evict a younger one, which stops two requests from handing the
pool back and forth.

**Cached prefixes are an allocation decision.** A request whose sequence starts with
tokens someone else already computed inherits those blocks. The scheduler plans the hit
like any other claim, with remaining work and block requirement both shrinking, and the
pool has to be able to fund the rest of the sequence before the share is taken, so
admission control and preemption keep working unchanged.

**Policies are one decision wide.** `fcfs`, `decode_first`, `prefill_first`, `balanced`
and `static` share the budget accounting, the memory planning and the eviction rule. They
differ in who gets the step, or for `static` who is allowed to join a batch.

---

## A real model in the same runtime

Everything above runs on a modelled execution cost, so that thousands of scheduling
experiments stay reproducible in seconds. The same scheduler, block manager and engine
also drive a real transformer with real KV tensors, which is what turns the block manager
from bookkeeping into memory a model attends over.

```bash
pip install -e ".[torch]"
python examples/real_model.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --prompt "Explain continuous batching in one sentence:" --max-new-tokens 24
```

```text
model  : HuggingFaceTB/SmolLM2-135M-Instruct
shape  : 30 layers, 9 heads (3 kv), head_dim 64, float32
kv pool: 128 blocks x 16 tokens = 90.0 MiB
budget : 64 tokens/step, 1 concurrent requests

── prompt 0: 'Explain continuous batching in one sentence:' (8 tokens)
[p0] "
[p0] The
[p0]  company
[p0] 's
[p0]  sales
[p0]  team
[p0]  is
[p0]  responsible
[p0]  for
...
[p0] 24/24 tokens: '\n\n"The company\'s sales team is responsible for managing the production of the products, which are then shipped to the'

steps=24 generated=24 throughput=75.1 tok/s ttft=39.9ms tpot=12.1ms kv_peak=1.6% evictions=0 wall=0.32s
```

An instruct checkpoint still continues the prompt rather than answering it, because the
demo feeds raw text and does no chat templating. Prompt formatting is a serving-layer
concern; this project is the scheduler and the cache underneath it.

KV lives in `[layers, blocks, kv_heads, block_size, head_dim]` tensors. Logical block *i*
of a request maps to a physical block through its block table, so the tensors attention
reads are the tensors the pool owns. Each step flattens every scheduled request's chunk
into one token sequence and runs one forward pass over it: per layer, the freshly computed
keys and values are written into their physical blocks, each request's cached context is
gathered back out by block table, and a block-diagonal causal mask keeps requests from
attending to each other.

I pinned this against HuggingFace's own forward pass rather than trusting the text, since
a wrong rotation or a truncated dtype still produces plausible tokens:

| Check | Result |
|---|---|
| Sampled logits vs `AutoModelForCausalLM` | max difference **6e-08** in float32 |
| The same, in float64 | max difference **5.6e-17** |
| Greedy tokens vs `model.generate` | identical |
| Chunked prefill, `block_size=1`, batching, preemption | identical tokens in every case |

The float64 row is the strong one: agreement to machine precision rules out a
mathematically different computation and leaves only float32 accumulation order.

A real forward samples the next token while computing the positions it was granted, so a
decoding request always carries exactly one position whose KV is still pending, and the
first token lands at the end of prefill, which is where a server first has something to
stream and therefore where TTFT belongs. And because a runner that attends over the pool
cannot compute before its blocks exist, the engine commits that runner's allocations
before it executes, while the simulated runner keeps the strictly-speculative order. Both
paths are covered by the check suite.

---

## What is measured

Definitions follow the ones vLLM publishes:

| Metric | Definition |
|---|---|
| TTFT | arrival → first generated token (includes queueing) |
| ITL | gap between consecutive streamed tokens, reported as a distribution |
| TPOT | `(end_to_end − TTFT) / (output_tokens − 1)`, per request |
| E2E | arrival → completion |
| Queue time | arrival → first step that ran the request |
| Throughput | generated tokens per second of modelled time |
| KV utilization | allocated blocks / total blocks, sampled per step |
| Internal fragmentation | reserved capacity holding no tokens, sampled per step |
| Decode batch | decoding requests per step |
| Prefill size | prefill tokens per step |

Every aggregate is traceable to the steps and tokens that produced it. Two runs of the
same policy on the same seed produce bit-identical metric rows, asserted by
`check_policies.py`, so a benchmark cannot be noise without a test failing.

---

## Limitations

| Limitation | Status |
|---|---|
| Execution cost is a stated linear model, not a GPU | deliberate: fast, deterministic, comparative |
| No fused paged-attention kernel | the real runner gathers blocks into contiguous tensors; correct, not fast |
| The real runner is CPU-only and small-model | no batching kernel, no CUDA, no tensor parallelism |
| Preemption recomputes; there is no swap path | recompute is simpler to reason about; swap is future work |
| `fcfs` and `decode_first` are the same schedule | tested and labelled a finding, not hidden |
| Only full blocks are cached | a partial tail block is the one a request is writing |
| Requests arriving together cannot share | nothing is cached until the first one computes it |
| Cached blocks still occupy the pool | they are evictable, but they compete for capacity |
| No priority classes or deadlines | ageing is the only fairness mechanism |
| A request that can never fit stalls the run | detected by `can_ever_fit()`; no rejection policy yet |
| Python-only, single process | no tensor/pipeline parallelism, on purpose |

---

## Reproduce

```bash
# headline table
python -m miniserve.cli --requests 120 --arrival poisson --rate 16

# every figure in this README, from a fixed seed
python scripts/make_figures.py                 # writes figures/*.png

# the full check suite (155 checks)
for f in check_engine check_memory check_preemption check_metrics check_benchmark \
         check_policies check_prefix_cache check_visualizations check_torch_runner; do
  python scripts/$f.py; done

# prefix caching, with and without
python examples/prefix_cache.py --requests 120 --rate 12
python -m miniserve.cli --sweep prefix_cache --values off,on --requests 120 --rate 12

# the real runner: paged attention, verified against HuggingFace
python examples/real_model.py --concurrency 3
python scripts/check_torch_runner.py
```

## Repository layout

```text
miniserve/
  engine/       request model, scheduling contract, policies, the execution loop
  memory/       KV block pool; BlockPlan is a transactional view of it
  runner/       execution-cost model, and the real torch runner with a paged KV pool
  metrics/      TTFT / ITL / TPOT / throughput / KV utilization / fragmentation
  benchmark/    arrival processes, workload generation, driver, sweeps, exports
  visualizations/  text timelines and charts, plus the matplotlib figures
scripts/        check_*.py (plain asserts) and make_figures.py
examples/       seven runnable demos, each showing one idea
figures/        generated by scripts/make_figures.py, committed for the README
```

**Reading order for the core logic** (~250 lines):
`engine/request.py` → `engine/scheduler.py` → `engine/policies.py` → `engine/engine.py`
→ `memory/block_manager.py`.

## Where this goes next

1. **Sharing across concurrent arrivals**: a request can only inherit a prefix that is
   already computed, so a burst of identical prompts still computes it several times. A
   "compute once, wait for it" admission rule would fix that.
2. **Swap-based preemption**: copy a victim's blocks to host memory instead of recomputing,
   and measure which wins as a function of prompt length, including when the prompt is
   still in the prefix cache.
3. **A fused paged-attention kernel**: the gather is the price of correctness and a kernel
   that reads blocks in place is the price of speed. Prefix caching makes that gather read
   the same blocks for many requests, which is what a kernel could exploit.

---

*Every number in this README came from a run on this machine. `TESTING.md` is the
companion hands-on guide: how to run each experiment, what to expect, which invariants to
poke at, and how to read the code.*
