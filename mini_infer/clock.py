"""Simulation and engine clock abstractions.

The engine never calls ``time`` directly. Simulated workloads run on a
:class:`VirtualClock` whose value is advanced by the modelled cost of each
engine step, which keeps large benchmarks fast and bit-for-bit reproducible.
The real model runner runs on :class:`WallClock` instead.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Source of monotonically increasing time in seconds."""

    def now(self) -> float: ...

    def advance(self, seconds: float) -> None: ...


class VirtualClock:
    """Deterministic clock driven by modelled execution costs."""

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"cannot advance clock backwards by {seconds}")
        self._now += float(seconds)

    def __repr__(self) -> str:
        return f"VirtualClock(now={self._now:.6f})"


class WallClock:
    """Real time, for the torch-backed runner."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._start

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"cannot advance clock backwards by {seconds}")
        time.sleep(seconds)

    def __repr__(self) -> str:
        return f"WallClock(now={self.now():.6f})"
