"""Workload generation: how requests arrive and how big they are.

Three arrival processes, because they test different things:

* **burst** — every request arrives at t=0. Nothing is a client here; the whole
  load is offered at once, so the engine's queue and KV pool are immediately under
  maximum pressure. This is a stress test, not a model of traffic.
* **poisson** — open-loop arrivals at a target rate. Requests arrive whether or not
  the engine is keeping up, so queueing delay grows without bound past saturation.
  This is what a load sweep varies.
* **closed_loop** — a fixed number of concurrent clients, each issuing its next
  request only once its previous one has finished. Offered load self-limits, so the
  queue never builds; this isolates scheduler behaviour from admission queueing.

Lengths come from a :class:`PromptProfile` and :class:`OutputProfile`, each of which
is a distribution plus a clamp. Every workload is derived from a seed, so a run can
be regenerated exactly.

The spec objects are dataclasses so a benchmark configuration can be serialised and
replayed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from mini_infer.engine.request import Request


class LengthDistribution(Protocol):
    """Draws a token count."""

    def sample(self, rng: random.Random) -> int: ...


@dataclass(frozen=True, slots=True)
class Constant:
    value: int

    def sample(self, rng: random.Random) -> int:
        return self.value


@dataclass(frozen=True, slots=True)
class Uniform:
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 1 or self.high < self.low:
            raise ValueError(f"invalid uniform range [{self.low}, {self.high}]")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Log-normal in token space: a long tail of large prompts, as real traffic has."""

    median: int
    sigma: float = 0.8

    def __post_init__(self) -> None:
        if self.median < 1:
            raise ValueError("median must be >= 1")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def sample(self, rng: random.Random) -> int:
        return max(1, round(rng.lognormvariate(math.log(self.median), self.sigma)))


@dataclass(frozen=True, slots=True)
class LengthProfile:
    """A token-count distribution with a hard clamp."""

    distribution: LengthDistribution
    minimum: int = 1
    maximum: int | None = None

    def sample(self, rng: random.Random) -> int:
        value = max(self.minimum, self.distribution.sample(rng))
        return value if self.maximum is None else min(value, self.maximum)


# Named profiles, so experiments read clearly and stay reproducible.
PROMPT_PROFILES: dict[str, LengthProfile] = {
    "short": LengthProfile(Constant(24)),
    "chat": LengthProfile(LogNormal(median=64, sigma=0.9), minimum=4, maximum=512),
    "mixed": LengthProfile(Uniform(32, 512)),
    "long_context": LengthProfile(LogNormal(median=1024, sigma=0.5), minimum=256, maximum=4096),
}

OUTPUT_PROFILES: dict[str, LengthProfile] = {
    "short": LengthProfile(Constant(16)),
    "balanced": LengthProfile(LogNormal(median=48, sigma=0.7), minimum=4, maximum=256),
    "long": LengthProfile(Uniform(64, 256)),
}


def prompt_profile(name: str) -> LengthProfile:
    try:
        return PROMPT_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown prompt profile {name!r}; available: {sorted(PROMPT_PROFILES)}"
        ) from None


def output_profile(name: str) -> LengthProfile:
    try:
        return OUTPUT_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown output profile {name!r}; available: {sorted(OUTPUT_PROFILES)}"
        ) from None


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """A reproducible description of the traffic to offer an engine."""

    num_requests: int
    prompt: LengthProfile = field(default_factory=lambda: PROMPT_PROFILES["chat"])
    output: LengthProfile = field(default_factory=lambda: OUTPUT_PROFILES["balanced"])
    arrival: str = "poisson"
    #: Requests per second for ``poisson``; ignored by the other processes.
    rate: float = 8.0
    #: Concurrent clients for ``closed_loop``.
    concurrency: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if self.arrival not in ("burst", "poisson", "closed_loop"):
            raise ValueError(
                f"unknown arrival {self.arrival!r}; "
                "available: ['burst', 'closed_loop', 'poisson']"
            )
        if self.arrival == "poisson" and self.rate <= 0:
            raise ValueError("rate must be > 0 for poisson arrivals")
        if self.arrival == "closed_loop" and self.concurrency < 1:
            raise ValueError("concurrency must be >= 1 for closed_loop arrivals")

    def build(self) -> Workload:
        """Materialise the workload, drawing every length from the seed."""
        rng = random.Random(self.seed)
        requests = [
            Request(
                id=f"req-{index}",
                prompt_tokens=list(range(self.prompt.sample(rng))),
                max_new_tokens=self.output.sample(rng),
                arrival_time=0.0,
            )
            for index in range(self.num_requests)
        ]
        runnable: ArrivalProcess
        if self.arrival == "burst":
            runnable = BurstArrivals()
        elif self.arrival == "poisson":
            runnable = PoissonArrivals(rate=self.rate, rng=random.Random(self.seed + 1))
        else:
            runnable = ClosedLoopArrivals(concurrency=self.concurrency)
        return Workload(spec=self, requests=requests, arrivals=runnable)


class ArrivalProcess(Protocol):
    """Decides when the next request is offered."""

    def next_arrival_time(
        self,
        *,
        previous_arrival: float,
        completed: int,
        in_flight: int,
        pending: int,
        num_requests: int,
    ) -> float | None:
        """Time of the next arrival, or None if this process has nothing more to offer.

        ``completed``, ``in_flight`` and ``pending`` describe the engine's view, which
        is what lets a closed-loop process wait for a completion. ``num_requests`` is
        the workload size, needed to tell a client pool that is full from one that has
        not started yet.
        """
        ...


@dataclass(slots=True)
class BurstArrivals:
    """Everything arrives at once: a pure stress test of the engine's queue."""

    def next_arrival_time(
        self,
        *,
        previous_arrival: float,
        completed: int,
        in_flight: int,
        pending: int,
        num_requests: int,
    ) -> float | None:
        return None if pending == 0 else 0.0


@dataclass(slots=True)
class PoissonArrivals:
    """Open-loop arrivals: the offered load does not care how busy the engine is."""

    rate: float
    rng: random.Random

    def next_arrival_time(
        self,
        *,
        previous_arrival: float,
        completed: int,
        in_flight: int,
        pending: int,
        num_requests: int,
    ) -> float | None:
        if pending == 0:
            return None
        if completed == 0 and in_flight == 0 and pending == num_requests:
            # Warm start: the first request is always visible at t=0.
            return 0.0
        return previous_arrival + self.rng.expovariate(self.rate)


@dataclass(slots=True)
class ClosedLoopArrivals:
    """A fixed pool of clients, each with at most one request in flight.

    A client whose request has finished issues its next request immediately, so the
    offered load self-limits and the engine's queue never builds. Time therefore makes
    no difference here: an arrival is either due now or blocked on a completion.
    """

    concurrency: int

    def next_arrival_time(
        self,
        *,
        previous_arrival: float,
        completed: int,
        in_flight: int,
        pending: int,
        num_requests: int,
    ) -> float | None:
        if pending == 0:
            return None
        # Clients that have not issued yet plus requests outstanding must not exceed
        # the pool. `submitted` counts what the driver has already offered.
        submitted = num_requests - pending
        busy = in_flight + (submitted - completed)
        return 0.0 if busy < self.concurrency else None


@dataclass(slots=True)
class Workload:
    """Requests plus the process that decides when they appear."""

    spec: WorkloadSpec
    requests: list[Request]
    arrivals: ArrivalProcess

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_len for r in self.requests)

    def total_output_tokens(self) -> int:
        return sum(r.max_new_tokens for r in self.requests)

    def describe(self) -> dict[str, object]:
        """A compact, serialisable description for a benchmark report."""
        prompts = [r.prompt_len for r in self.requests]
        outputs = [r.max_new_tokens for r in self.requests]
        return {
            "num_requests": self.num_requests,
            "arrival": self.spec.arrival,
            "rate": self.spec.rate,
            "concurrency": self.spec.concurrency,
            "seed": self.spec.seed,
            "prompt_tokens": sum(prompts),
            "output_tokens": sum(outputs),
            "mean_prompt": round(sum(prompts) / len(prompts), 2),
            "max_prompt": max(prompts),
            "mean_output": round(sum(outputs) / len(outputs), 2),
            "max_output": max(outputs),
        }
