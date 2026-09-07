"""Model runner contract: what actually performs prefill and decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from mini_infer.engine.scheduler import SchedulerOutput


@dataclass(slots=True)
class RunnerResult:
    """Output token produced for each decoding request in a step.

    A simulated runner fills this with placeholder ids; a real runner fills it
    with sampled token ids.
    """

    sampled_tokens: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimingResult:
    """How long the runner says a step took."""

    duration_s: float
    prefill_cost_s: float = 0.0
    decode_cost_s: float = 0.0


class ModelRunner(Protocol):
    """Executes one :class:`SchedulerOutput`."""

    def execute(self, output: SchedulerOutput, *, context_lengths: dict[str, int]) -> RunnerResult:
        """Perform the scheduled work and return sampled tokens.

        ``context_lengths`` is the sequence length each scheduled request will
        hold once the step completes, so runners can size attention cost.
        """
        ...

    def time_step(
        self, output: SchedulerOutput, *, context_lengths: dict[str, int]
    ) -> TimingResult:
        """Report the modelled (or measured) duration of the step."""
        ...
