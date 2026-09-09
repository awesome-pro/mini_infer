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
    """Executes one :class:`SchedulerOutput`.

    Two runner styles are supported, distinguished by ``attends_over_kv``:

    * A **simulated** runner models the cost and returns placeholder tokens. It needs
      nothing from the KV pool, so the engine commits allocations after it returns and
      the sequence advances in lockstep with the cursor.
    * A runner that **attends over the KV blocks** cannot compute without them, so the
      engine commits its allocations before it runs, and it samples a token for every
      request it is given. See :attr:`attends_over_kv`.
    """

    #: Set by runners that read and write the engine's physical KV blocks. Such a
    #: runner needs its blocks committed *before* it executes, and it returns a
    #: sampled token for every scheduled request rather than only for decodes.
    attends_over_kv: bool = False

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
