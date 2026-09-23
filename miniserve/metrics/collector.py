"""Metrics: what the runtime actually did, measured rather than asserted.

A runtime is not finished when its output is "all requests completed". This module
turns an engine run into the numbers serving systems are judged on: TTFT, ITL,
TPOT, end-to-end latency, queue time, throughput and KV pressure.

Definitions follow the ones vLLM's benchmark tooling publishes, so the numbers are
comparable with how real engines report:

* **TTFT**  arrival to the first streamed output token
* **TPOT**  per request, ``(end_to_end - ttft) / (output_tokens - 1)``: the mean
  time per output token once streaming has begun
* **ITL**   the gap between consecutive streamed tokens, so its distribution shows
  the stalls that a mean hides
* **E2E**   arrival to completion
* **queue time**  arrival to the first step that ran the request
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from miniserve.engine.request import Request

if TYPE_CHECKING:
    from miniserve.engine.engine import Engine, EngineStep


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile, matching numpy's default definition.

    Reported alongside the mean everywhere, because latency distributions in a
    serving system are long-tailed: a mean TTFT hides the requests that waited.
    """
    if not values:
        return float("nan")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(slots=True)
class RequestMetrics:
    """Everything measured about one request, plus its token timeline."""

    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_time: float
    first_token_time: float | None = None
    finish_time: float | None = None
    first_step: int | None = None
    first_step_time: float | None = None
    num_preemptions: int = 0
    cached_prefix_tokens: int = 0
    #: Wall time of every streamed token, which is what makes an ITL distribution
    #: possible instead of only an average.
    token_times: list[float] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.finish_time is not None

    @property
    def ttft(self) -> float:
        """Arrival to first token; NaN while no token has streamed."""
        first = self.first_token_time
        if first is None and self.token_times:
            first = self.token_times[0]
        if first is None:
            return float("nan")
        return first - self.arrival_time

    @property
    def e2e(self) -> float:
        """Arrival to completion; NaN while unfinished."""
        if self.finish_time is None:
            return float("nan")
        return self.finish_time - self.arrival_time

    @property
    def queue_time(self) -> float:
        """Arrival to the first step that ran this request.

        Includes admission delay from the running-request limit or KV pressure, so
        it separates "the engine was busy" from "generation was slow".
        """
        if self.first_step_time is None:
            return float("nan")
        return self.first_step_time - self.arrival_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Gaps between consecutive streamed tokens."""
        return [
            later - earlier
            for earlier, later in zip(self.token_times, self.token_times[1:], strict=False)
        ]

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first, per vLLM's definition."""
        if not self.finished or self.output_tokens < 2:
            return float("nan")
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


@dataclass(slots=True)
class StepMetrics:
    """What one engine step looked like."""

    index: int
    duration: float
    num_scheduled_tokens: int
    num_prefill_tokens: int
    num_decode_requests: int
    running_requests: int
    waiting_requests: int
    kv_free_blocks: int | None
    kv_total_blocks: int | None
    num_preemptions: int
    internal_fragmentation: float = 0.0

    @property
    def kv_utilization(self) -> float:
        if not self.kv_total_blocks or self.kv_free_blocks is None:
            return 0.0
        return (self.kv_total_blocks - self.kv_free_blocks) / self.kv_total_blocks


class MetricsCollector:
    """Records an engine run and reports its latency and throughput.

    Attached to an engine, so it observes the engine's own clock. Under a
    :class:`~miniserve.clock.VirtualClock` that clock is the modelled execution
    time, which is what makes simulated benchmarks reproducible; under a
    ``WallClock`` it is real elapsed time.
    """

    def __init__(self) -> None:
        self.requests: dict[str, RequestMetrics] = {}
        self.steps: list[StepMetrics] = []
        #: Prefix-cache reuse: how many requests inherited KV, and how many tokens.
        self.prefix_cache_hits = 0
        self.cached_prefix_tokens = 0
        self._engine: Engine | None = None

    # ------------------------------------------------------------- attachment

    def attach(self, engine: Engine) -> None:
        self._engine = engine

    # -------------------------------------------------------------- recording

    def register(self, request: Request) -> None:
        """Start tracking a request, usually at submission."""
        if request.id in self.requests:
            return
        self.requests[request.id] = RequestMetrics(
            request_id=request.id,
            prompt_tokens=request.prompt_len,
            output_tokens=request.max_new_tokens,
            arrival_time=request.arrival_time,
        )

    def _metrics_for(self, request: Request) -> RequestMetrics:
        if request.id not in self.requests:
            self.register(request)
        return self.requests[request.id]

    def record_step(self, step: EngineStep) -> None:
        """Record one engine step and every request-level event inside it."""
        from miniserve.engine.engine import StepEventKind

        for event in step.events:
            metrics = self._metrics_for_id(event.request_id)
            if event.kind is StepEventKind.DECODED:
                metrics.token_times.append(step.end_time)
                if metrics.first_token_time is None:
                    metrics.first_token_time = step.end_time
            elif event.kind is StepEventKind.PREEMPTED:
                metrics.num_preemptions += 1
            elif event.kind is StepEventKind.PREFIX_CACHED:
                self.prefix_cache_hits += 1
                self.cached_prefix_tokens += event.num_tokens
                metrics.cached_prefix_tokens += event.num_tokens
            elif event.kind is StepEventKind.FINISHED:
                metrics.finish_time = step.end_time

            # Queue time ends at the first step that ran the request, whether that
            # was its admission or a recomputation after preemption.
            if metrics.first_step_time is None and event.kind in (
                StepEventKind.ADMITTED,
                StepEventKind.DECODED,
            ):
                metrics.first_step = step.index
                metrics.first_step_time = step.start_time

        self.steps.append(
            StepMetrics(
                index=step.index,
                duration=step.duration,
                num_scheduled_tokens=step.output.num_scheduled_tokens,
                num_prefill_tokens=step.num_prefill_tokens,
                num_decode_requests=step.num_decode_requests,
                running_requests=step.running_requests,
                waiting_requests=step.waiting_requests,
                kv_free_blocks=step.free_blocks,
                kv_total_blocks=step.total_blocks,
                num_preemptions=step.num_preemptions,
                internal_fragmentation=self._fragmentation(),
            )
        )

    def _fragmentation(self) -> float:
        """Tail waste in the KV pool right now: reserved capacity holding no tokens.

        Sampled per step because it is only meaningful while requests are resident:
        once a run finishes, every block is free and the final state reads as zero.
        """
        memory = self._engine.memory if self._engine is not None else None
        if memory is None:
            return 0.0
        return memory.stats().internal_fragmentation

    def _metrics_for_id(self, request_id: str) -> RequestMetrics:
        if request_id not in self.requests:
            self.requests[request_id] = RequestMetrics(
                request_id=request_id,
                prompt_tokens=0,
                output_tokens=0,
                arrival_time=0.0,
            )
        return self.requests[request_id]

    def record_finish(self, request: Request, finish_time: float) -> None:
        """Record completion for a request the step events did not cover."""
        metrics = self._metrics_for(request)
        if metrics.finish_time is None:
            metrics.finish_time = finish_time

    # ----------------------------------------------------------------- report

    def summary(self, *, elapsed: float) -> RunMetrics:
        return RunMetrics.from_collector(self, elapsed=elapsed)


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate results for one engine run."""

    num_requests: int
    num_finished: int
    num_preemptions: int
    num_steps: int
    elapsed_time: float

    prompt_tokens: int
    generated_tokens: int

    mean_ttft: float
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float

    mean_tpot: float
    p50_tpot: float
    p95_tpot: float

    mean_itl: float
    p50_itl: float
    p95_itl: float
    p99_itl: float

    mean_e2e: float
    p50_e2e: float
    p95_e2e: float

    mean_queue_time: float
    p95_queue_time: float

    peak_kv_utilization: float
    mean_kv_utilization: float
    mean_internal_fragmentation: float
    max_internal_fragmentation: float
    prefix_cache_hits: int
    cached_prefix_tokens: int

    mean_prefill_tokens_per_step: float
    max_prefill_tokens_per_step: int
    mean_decode_batch: float
    max_decode_batch: int

    @property
    def output_throughput(self) -> float:
        """Generated tokens per second over the whole run."""
        return self.generated_tokens / self.elapsed_time if self.elapsed_time > 0 else 0.0

    @property
    def total_throughput(self) -> float:
        """Prompt plus generated tokens per second."""
        if self.elapsed_time <= 0:
            return 0.0
        return (self.prompt_tokens + self.generated_tokens) / self.elapsed_time

    @property
    def requests_per_second(self) -> float:
        return self.num_finished / self.elapsed_time if self.elapsed_time > 0 else 0.0

    @classmethod
    def from_collector(cls, collector: MetricsCollector, *, elapsed: float) -> RunMetrics:
        finished = [m for m in collector.requests.values() if m.finished]
        ttfts = [m.ttft for m in finished if not math.isnan(m.ttft)]
        tpots = [m.tpot for m in finished if not math.isnan(m.tpot)]
        e2es = [m.e2e for m in finished]
        queues = [m.queue_time for m in finished if not math.isnan(m.queue_time)]
        itls = [gap for m in finished for gap in m.inter_token_latencies]

        prefill_per_step = [s.num_prefill_tokens for s in collector.steps]
        decode_per_step = [s.num_decode_requests for s in collector.steps]
        kv = [s.kv_utilization for s in collector.steps]
        fragmentation = [s.internal_fragmentation for s in collector.steps]

        return cls(
            num_requests=len(collector.requests),
            num_finished=len(finished),
            num_preemptions=sum(m.num_preemptions for m in collector.requests.values()),
            num_steps=len(collector.steps),
            elapsed_time=elapsed,
            prompt_tokens=sum(m.prompt_tokens for m in collector.requests.values()),
            generated_tokens=sum(len(m.token_times) for m in collector.requests.values()),
            mean_ttft=_mean(ttfts),
            p50_ttft=percentile(ttfts, 0.50),
            p95_ttft=percentile(ttfts, 0.95),
            p99_ttft=percentile(ttfts, 0.99),
            mean_tpot=_mean(tpots),
            p50_tpot=percentile(tpots, 0.50),
            p95_tpot=percentile(tpots, 0.95),
            mean_itl=_mean(itls),
            p50_itl=percentile(itls, 0.50),
            p95_itl=percentile(itls, 0.95),
            p99_itl=percentile(itls, 0.99),
            mean_e2e=_mean(e2es),
            p50_e2e=percentile(e2es, 0.50),
            p95_e2e=percentile(e2es, 0.95),
            mean_queue_time=_mean(queues),
            p95_queue_time=percentile(queues, 0.95),
            peak_kv_utilization=max(kv) if kv else 0.0,
            mean_kv_utilization=_mean(kv),
            mean_internal_fragmentation=_mean(fragmentation),
            max_internal_fragmentation=max(fragmentation) if fragmentation else 0.0,
            prefix_cache_hits=collector.prefix_cache_hits,
            cached_prefix_tokens=collector.cached_prefix_tokens,
            mean_prefill_tokens_per_step=_mean(prefill_per_step),
            max_prefill_tokens_per_step=max(prefill_per_step) if prefill_per_step else 0,
            mean_decode_batch=_mean(decode_per_step),
            max_decode_batch=max(decode_per_step) if decode_per_step else 0,
        )

    def as_row(self) -> dict[str, float | int]:
        """Flat dict, convenient for CSV and for building a results table."""
        return {
            "requests": self.num_requests,
            "finished": self.num_finished,
            "preemptions": self.num_preemptions,
            "steps": self.num_steps,
            "elapsed_s": round(self.elapsed_time, 4),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "output_tok_per_s": round(self.output_throughput, 2),
            "total_tok_per_s": round(self.total_throughput, 2),
            "req_per_s": round(self.requests_per_second, 3),
            "mean_ttft_ms": round(self.mean_ttft * 1e3, 3),
            "p50_ttft_ms": round(self.p50_ttft * 1e3, 3),
            "p95_ttft_ms": round(self.p95_ttft * 1e3, 3),
            "p99_ttft_ms": round(self.p99_ttft * 1e3, 3),
            "mean_tpot_ms": round(self.mean_tpot * 1e3, 3),
            "p95_tpot_ms": round(self.p95_tpot * 1e3, 3),
            "mean_itl_ms": round(self.mean_itl * 1e3, 3),
            "p50_itl_ms": round(self.p50_itl * 1e3, 3),
            "p95_itl_ms": round(self.p95_itl * 1e3, 3),
            "p99_itl_ms": round(self.p99_itl * 1e3, 3),
            "mean_e2e_ms": round(self.mean_e2e * 1e3, 3),
            "p95_e2e_ms": round(self.p95_e2e * 1e3, 3),
            "mean_queue_ms": round(self.mean_queue_time * 1e3, 3),
            "p95_queue_ms": round(self.p95_queue_time * 1e3, 3),
            "peak_kv_utilization": round(self.peak_kv_utilization, 4),
            "mean_kv_utilization": round(self.mean_kv_utilization, 4),
            "mean_fragmentation": round(self.mean_internal_fragmentation, 4),
            "max_fragmentation": round(self.max_internal_fragmentation, 4),
            "prefix_cache_hits": self.prefix_cache_hits,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "mean_prefill_tokens": round(self.mean_prefill_tokens_per_step, 2),
            "max_prefill_tokens": self.max_prefill_tokens_per_step,
            "mean_decode_batch": round(self.mean_decode_batch, 2),
            "max_decode_batch": self.max_decode_batch,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
