# **Mini Inference Runtime**

The objective is **not**:

> “I wrote a Python scheduler that moves requests between queues.”

The objective is:

> **I built a miniature LLM serving runtime that accepts concurrent generation requests, schedules prefill/decode under compute and KV-memory constraints, manages block-based KV memory, supports continuous batching and chunked prefill, and benchmarks the latency/throughput trade-offs of different scheduling policies.**

That is a serious project.

Modern vLLM itself makes scheduling decisions at the engine-step level: each step corresponds roughly to one model forward pass, and the scheduler decides how many tokens each request gets in that step. That can be a full prompt, one decode token, or a partial prefill chunk. ([vLLM][1])

# 1. What the finished system should actually do

Imagine these requests reach your server:

```text
t=0
R1: prompt=80 tokens,  max_new_tokens=20
R2: prompt=12 tokens,  max_new_tokens=30

t=1
R3: prompt=300 tokens, max_new_tokens=10

t=3
R4: prompt=8 tokens,   max_new_tokens=50
```

Suppose your engine has:

```text
max_running_requests = 4
max_batch_tokens      = 64
KV blocks             = 32
block_size            = 16 tokens
```

Your engine should continuously decide:

```text
Engine step 1
R1 → prefill 52 tokens
R2 → prefill 12 tokens

Engine step 2
R1 → prefill remaining 28
R2 → decode 1 token
R3 → prefill 35 tokens

Engine step 3
R1 → decode 1
R2 → decode 1
R3 → prefill 61

...
```

While doing this, it also needs to:

```text
allocate KV blocks
release KV blocks
track request state
respect token budget
prevent decode latency from exploding
prevent prefills from starving forever
admit new requests continuously
measure TTFT / ITL / throughput
```

**That is the core project.**

---

# 2. Final architecture

I would design it like this:

```text
                  ┌─────────────────────┐
Requests ────────▶│    Request Queue    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      Scheduler      │
                  │                     │
                  │ token budget        │
                  │ continuous batching │
                  │ chunked prefill     │
                  │ decode priority     │
                  │ preemption/fairness │
                  └──────────┬──────────┘
                             │
                  SchedulerOutput
                  {
                    r1: 1,
                    r2: 1,
                    r3: 32
                  }
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │ KV Block Manager │          │   Model Runner   │
    │                  │          │                  │
    │ allocate blocks  │          │ prefill          │
    │ free blocks      │          │ decode           │
    │ block tables     │          │ produce tokens   │
    │ cache pressure   │          │                  │
    └────────┬─────────┘          └────────┬─────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                 ┌─────────────────────┐
                 │       Engine        │
                 │ update request state│
                 │ collect output      │
                 │ repeat next step    │
                 └──────────┬──────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ Metrics / Benchmark   │
                │                       │
                │ TTFT                  │
                │ TPOT / ITL            │
                │ throughput            │
                │ queue latency         │
                │ KV utilization        │
                └───────────────────────┘
```

This separation is important.

Your **Scheduler decides what should run**.

Your **KV manager decides whether the memory exists to run it**.

Your **ModelRunner actually performs the work**.

Your **Engine coordinates everything**.

---

# 3. The six important components

| Component          | Responsibility                                        |
| ------------------ | ----------------------------------------------------- |
| `Request`          | Represents one generation request and its state       |
| `Scheduler`        | Determines what executes in the next engine iteration |
| `KVBlockManager`   | Allocates/reclaims KV cache capacity                  |
| `ModelRunner`      | Performs prefill/decode                               |
| `Engine`           | Main execution loop                                   |
| `MetricsCollector` | Measures system behavior                              |

I would roughly structure the repo as:

```text
miniserveence-engine/

  engine/
    request.py
    scheduler.py
    engine.py

  memory/
    block.py
    block_manager.py

  runner/
    simulated_runner.py
    torch_runner.py

  metrics/
    collector.py

  benchmark/
    workloads.py
    benchmark.py

  visualizations/
    timeline.py
    plots.py

  tests/
    test_scheduler.py
    test_block_manager.py
    test_engine.py

  examples/
    demo.py

  README.md
```

---

# 4. Request model

Keep this extremely understandable.

```python
@dataclass
class Request:
    id: str

    prompt_tokens: list[int]
    max_new_tokens: int

    arrival_time: float

    status: RequestStatus

    prefilled_tokens: int = 0
    generated_tokens: list[int] = field(default_factory=list)

    block_table: list[int] = field(default_factory=list)

    first_token_time: float | None = None
    finish_time: float | None = None
```

States:

```text
WAITING
   ↓
PREFILLING
   ↓
DECODING
   ↓
FINISHED
```

Eventually:

```text
WAITING
PREFILLING
DECODING
PREEMPTED
FINISHED
```

Don't create 15 states.

---

# 5. Scheduler: the heart of the project

This is where most of the signal comes from.

Its interface should roughly be:

```python
scheduler.schedule() -> SchedulerOutput
```

And:

```python
SchedulerOutput(
    scheduled_tokens={
        "r1": 1,
        "r2": 1,
        "r3": 30,
    }
)
```

Notice something important.

The scheduler doesn't say:

```text
prefill r3
```

It says:

```text
give r3 30 compute tokens
```

That's a much more general abstraction.

Current vLLM uses essentially this idea: per scheduling step it determines a token count for each request, where that token count could represent prompt computation, decode, chunked prefill, prefix-cache situations, or other execution modes. ([vLLM][1])

---

# 6. Start with continuous batching

First establish the difference between:

```text
STATIC BATCHING

R1 ────────────────────── FINISH
R2 ───────── FINISH
                         ↓
                wait for R1
                         ↓
R3 ─────────────────────────────
```

versus:

```text
CONTINUOUS BATCHING

R1 ──────────────────────────
R2 ─────── FINISH
          ↓ slot freed

R3        ────────────────────
```

As soon as a request finishes, another request can join the active set.

You've already implemented much of this conceptually, so don't rebuild it unnecessarily.

---

# 7. Add a token budget

This is where the scheduler starts becoming interesting.

Have:

```python
max_batch_tokens = 64
```

Every engine iteration has at most:

```text
64 tokens of compute
```

Suppose:

```text
R1: decoding
R2: decoding
R3: 200 prefill tokens remaining
```

You could schedule:

```text
R1 → 1
R2 → 1
R3 → 62

Total = 64
```

This is similar to a major control in production serving systems: current vLLM exposes `max_num_batched_tokens`, described as the maximum tokens processed within an iteration. ([vLLM][2])

This variable becomes the central scheduling resource in your engine.

---

# 8. Implement chunked prefill

Now suppose:

```text
R3 prompt = 500 tokens

batch token budget = 64
```

Do **not** require all 500 tokens to fit.

Instead:

```text
step 1 → 64
step 2 → 64
step 3 → 64
...
```

But when decode requests exist:

```text
R1 decode → 1
R2 decode → 1
R3 prefill → 62
```

That is chunked prefill.

Current vLLM explicitly supports chunking prefills according to the remaining batched-token budget. ([vLLM][2])

Why does this matter?

Because a giant prompt shouldn't necessarily monopolize an entire execution step and create huge latency spikes for requests that are already decoding.

Official vLLM performance documentation describes exactly this trade-off: smaller prefill chunks can reduce interruptions to decode/ITL, while chunks that are too small introduce overhead and can reduce throughput. ([vLLM][3])

That gives you a **great experiment** later.

---

# 9. Decode scheduling should become a policy

Don't hard-code one behavior.

Create:

```python
class SchedulingPolicy(Protocol):
    def schedule(...):
        ...
```

Then compare:

```text
FCFS
Prefill-first
Decode-first
Latency-aware / balanced
```

Your balanced policy might roughly do:

```python
budget = 64

# give existing decoding requests their next token
for req in decoding_requests:
    schedule(req, 1)
    budget -= 1

# use remaining compute for prefill
for req in waiting_or_prefilling:
    chunk = min(
        remaining_prefill(req),
        max_prefill_chunk,
        budget,
    )

    schedule(req, chunk)
    budget -= chunk
```

But then you discover a problem.

What happens with:

```text
70 decoding requests
budget = 64
```

Prefills might never run.

Excellent.

Now you've discovered **starvation**.

Add fairness through something like:

```text
max_wait_steps
prefill_reservation
age-based priority
periodic prefill admission
```

That is precisely the kind of engineering trade-off interviewers can discuss with you.

Don't hide it.

**Benchmark it.**

---

# 10. Build a real KV block manager

This is the second-biggest part of the project.

Do not represent KV capacity as:

```python
used_tokens += 1
```

Build actual blocks.

For example:

```text
block_size = 16 tokens

Physical KV blocks:

0
1
2
3
4
5
...
```

R1 might own:

```text
logical sequence blocks

R1 logical block 0 → physical block 7
R1 logical block 1 → physical block 2
R1 logical block 2 → physical block 19
```

Its block table becomes:

```python
R1.block_table = [7, 2, 19]
```

The blocks **do not need to be contiguous**.

That's the important concept.

vLLM's current paged KV implementation similarly maps logical blocks/token positions to physical KV blocks. ([vLLM][4])

Your manager should conceptually expose:

```python
allocate(req, num_tokens)

append_slot(req)

free(req)

can_allocate(req, num_tokens)

num_free_blocks()

utilization()
```

---

# 11. Show why block-based KV allocation exists

This part is valuable for your README.

Without blocks you might reserve:

```text
prompt + max_new_tokens
```

for every request.

Example:

```text
Request asks for maximum 2048 output tokens.

Actually generates 67.
```

Large amounts of reserved capacity were unnecessary.

With block-based growth:

```text
16 tokens generated → block
next 16 → another block
next 16 → another block
```

Memory grows with the sequence.

Your project should visually show:

```text
KV pool

[ R1 ][ R3 ][ FREE ][ R1 ][ R2 ][ FREE ][ R4 ][ R3 ]
```

This visualization will be much stronger than 500 lines of allocator code.

Important resume wording: unless you actually implement attention over paged physical tensors, call this a **block-based / paged-style KV cache manager**, not a full implementation of PagedAttention.

---

# 12. Add KV-memory pressure

Now make things difficult.

Suppose:

```text
total KV blocks = 20

free blocks = 1

R7 needs 4
```

The scheduler must decide:

```text
cannot admit R7
```

or later:

```text
preempt something
```

Version 1:

```text
WAIT
```

Version 2:

implement preemption.

For example:

```text
R1 uses 10 blocks
R2 uses 8
R3 waiting

KV exhausted
```

You can preempt a lower priority/younger request:

```text
R2 → PREEMPTED

free R2's KV
admit R3
```

Later:

```text
R2 → resume/recompute
```

The important thing is demonstrating that **compute scheduling and memory scheduling are coupled**.

That's a major inference-engine insight.

---

# 13. Have two ModelRunner implementations

This is something I strongly recommend.

### `SimulatedModelRunner`

This is for experiments.

Instead of loading a GPU model, estimate execution time from scheduled work:

```python
prefill_cost = f(number_of_prefill_tokens)
decode_cost = f(batch_size, context_lengths)
```

Why?

Because now you can run:

```text
10,000 requests
20 scheduling configurations
different KV capacities
different prompt distributions
```

within seconds.

And produce deterministic experiments.

### `TorchModelRunner`

Then add a real model.

Use something small:

```text
SmolLM
Qwen small model
TinyLlama
GPT-2 class model
```

The purpose isn't benchmark leadership.

The purpose is:

```text
curl /generate

→ request enters scheduler
→ prefill happens
→ KV state created
→ decode iterations happen
→ tokens stream back
```

So you can demonstrate:

**this isn't merely a simulator.**

---

# 14. Don'build a high-performance attention kernel yet

You don't need to implement:

```text
FlashAttention
CUDA kernels
tensor parallelism
NCCL
distributed inference
custom quantization kernels
```

Those belong in another project.

Trying to implement everything will turn a sharp scheduler project into a messy miniature-vLLM clone.

---

# 15. Metrics are mandatory

Your project is not finished if the only output is:

```text
all requests completed successfully
```

Production vLLM exposes both request-level and engine-level observability, including TTFT, ITL, TPOT, queue time, end-to-end latency, running request counts and KV-cache utilization. ([vLLM][5])

Your engine should collect at least:

| Metric                 | Meaning                          |
| ---------------------- | -------------------------------- |
| **TTFT**               | arrival → first generated token  |
| **ITL**                | time between streamed tokens     |
| **TPOT**               | average decode time/output token |
| **E2E latency**        | arrival → completion             |
| **Queue time**         | arrival → first scheduling       |
| **Throughput**         | tokens/sec                       |
| **KV utilization**     | allocated blocks / total blocks  |
| **Active requests**    | number currently running         |
| **Prefill batch size** | prefill tokens per iteration     |
| **Decode batch size**  | decode requests per iteration    |

vLLM's benchmark tooling defines TTFT as request-send to first streamed output and TPOT per request as approximately `(end-to-end latency - TTFT) / (output tokens - 1)`. ([vLLM][6])

Follow those definitions.

That makes your numbers comparable to terminology used in real serving systems.

---

# 16. The killer part: experiments

This is what will turn the repo from **“student implementation”** into **“engineering artifact.”**

I want your README to eventually contain experiments like this:

### Experiment A: Static vs continuous batching

```text
Workload:
100 requests
random arrivals
prompt length 32–512
output length 16–128

                  Throughput    Mean TTFT    P95 latency

Static              900          420ms        2.8s
Continuous         1250          260ms        1.9s
```

Don't fabricate these numbers, obviously. Measure them.

### Experiment B: Chunked vs unchunked prefill

Use:

```text
one 4K-token prompt
twenty short decoding requests
```

Show:

```text
                     P95 ITL       TTFT long request

No chunking             ↑                 ↓
256-token chunks        ↓                 ↑
64-token chunks         ↓↓                ↑↑
```

Now you can explain the tradeoff.

### Experiment C: Prefill-first vs decode-first

Show:

```text
decode-first

good ITL
bad waiting-request TTFT

prefill-first

good TTFT
decode latency spikes
```

Then introduce your balanced scheduler.

### Experiment D: KV pressure

Change:

```text
16 blocks
32 blocks
64 blocks
128 blocks
```

Measure:

```text
queueing
preemptions
throughput
KV utilization
```

### Experiment E: block size

Compare:

```text
8
16
32
64 tokens/block
```

Measure internal fragmentation.

This one is particularly nice because now the interviewer can ask:

> Why did you choose block size 16?

and you have an experimental answer.

---

# 17. Build a scheduler timeline visualizer

This would make the repository **much easier to understand visually**.

Something like:

```text
             step 1   step 2   step 3   step 4   step 5

R1           P64      P36      D1       D1       D1
R2           -        P20      D1       D1       D1
R3           -        -        P61      P64      P55
R4           -        -        D1       D1       D1
```

Where:

```text
P64 = 64 prefill tokens
D1  = one decode token
```

And ideally render a Gantt chart.

Separately graph:

```text
KV utilisation over time
running/waiting requests
TTFT distribution
ITL distribution
throughput
```

This will make your project far easier for a hiring manager to inspect in 60 seconds.

---

# 18. Add a small serving interface at the end

Eventually:

```bash
python server.py
```

Then:

```http
POST /generate

{
  "prompt": "Explain continuous batching",
  "max_new_tokens": 100
}
```

and stream:

```text
data: Continuous
data: batching
data: allows
...
```

You don't need OpenAI compatibility initially.

But if it's cheap to add later:

```text
/v1/completions
```

nice.

The **API is not the project**.

The runtime behind it is.

---

# 19. Tests matter a lot here

This is one place where using your coding agent heavily makes sense.

You should have invariants such as:

```text
allocated blocks + free blocks == total blocks

no physical block belongs to two requests

finished requests own zero blocks

scheduled tokens <= token budget

decode request gets <= 1 normal decode token/step

prefilled_tokens <= prompt_tokens

generated_tokens <= max_new_tokens

no waiting request remains forever under fairness policy
```

Property-based tests would actually be interesting here.

Throw random request sequences at the scheduler and verify these invariants.

That's a nice engineering signal.

---

# 20. What I would make optional

After the core system works, these become stretch features:

| Feature                   |                   Value |
| ------------------------- | ----------------------: |
| Prefix caching            |               Very high |
| Priority scheduling       |                  Medium |
| Preemption/recomputation  |                    High |
| Speculative decoding      |   High but larger scope |
| Prometheus `/metrics`     |                    Nice |
| Real GPU KV block tensors | Very high but difficult |
| OpenAI-compatible API     |                    Nice |
| Multi-GPU                 |        **Don't do now** |
| Tensor parallelism        |    **Separate project** |
| CUDA kernels              |    **Separate project** |

Prefix caching would be my favourite stretch feature.

You could hash full prompt blocks:

```text
"The capital of France is..."
        ↓
hash(block)
        ↓
KV block already exists?
```

Then reuse cached blocks for shared prefixes.

But only after everything else works.

---

# 21. The final demo should be easy to understand for the someone trying

Someone opens the GitHub repository.

They immediately see:

```text
MiniServe
A miniature LLM inference runtime implementing continuous batching,
chunked prefill, token-budget scheduling and block-based KV management.
```

Then an architecture image.

Then:

```bash
git clone ...
python benchmark.py --policy continuous
```

Then:

```text
Requests:              1000
Prompt tokens:         302,211
Generated tokens:       63,881

Throughput:              1,284 tok/s
Mean TTFT:                 181 ms
P95 TTFT:                  417 ms
Mean TPOT:                18.2 ms
P95 ITL:                  31.7 ms
KV peak utilization:      87.4%
```

Then charts.

Then:

```text
Static vs Continuous
Chunked vs Unchunked
Prefill-first vs Decode-first
KV pressure experiments
```

Then the architecture explanation.

Then source code.

That's what makes it impressive.

---

# 22. Definition of done

I would consider the project genuinely resume-ready when these **ten things** are true:

1. Requests arrive dynamically over time.
2. Continuous batching works.
3. Every engine iteration operates under a token budget.
4. Prefill can be chunked.
5. Decode requests are interleaved with prefills.
6. KV memory is managed using fixed physical blocks/block tables.
7. Memory is reclaimed when requests finish.
8. At least two scheduling policies can be compared.
9. TTFT, ITL/TPOT, throughput and KV utilization are measured.
10. README contains **real experiments and conclusions**, not just feature descriptions.

A real small-model backend is a strong addition, but I would **not block the first resume version on it** if the simulator/runtime architecture and experiments are already excellent.

---

# 23. How I would build it from where you currently are

I would build this roadmap:

| Phase       | Build                                 | Result                       |
| ----------- | ------------------------------------- | ---------------------------- |
| **1**       | Clean Request + Engine abstractions   | proper runtime skeleton      |
| **2**       | Continuous batching + token budget    | basic scheduler              |
| **3**       | Chunked prefill + decode interleaving | realistic scheduler          |
| **4**       | Physical KV block manager             | memory-aware serving         |
| **5**       | KV pressure + admission/preemption    | scheduler/memory interaction |
| **6**       | Metrics collector                     | TTFT/ITL/throughput          |
| **7**       | Workload simulator                    | repeatable benchmarking      |
| **8**       | Policy comparison                     | meaningful experiments       |
| **9**       | Visualizations + README               | portfolio artifact           |
| **10**      | Small real-model runner               | actual generation demo       |
| **Stretch** | Prefix cache                          | stronger inference depth     |

And there is one thing I particularly **do not want us to do**:

have the coding agent generate this whole repository in one go.

Instead, we should build it subsystem by subsystem.

### The final resume bullet could eventually become something like:

> **Mini LLM Inference Runtime** — Built a serving runtime with continuous batching, token-budget scheduling, chunked prefill and block-based KV-cache management; benchmarked scheduling policies across mixed workloads using TTFT, TPOT, throughput and KV utilization.

And once we have real numbers:

> Improved P95 inter-token latency by **X%** under mixed prefill/decode workloads while maintaining **Y tok/s** throughput compared with a naïve FCFS scheduler.
