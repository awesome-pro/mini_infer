"""A runner that models execution cost instead of executing a model.

This exists so that scheduling policy can be studied over tens of thousands of
requests in seconds, deterministically. The cost model is a stated
approximation (a roofline-style linear form), not a measurement of any GPU.
"""

from __future__ import annotations

from mini_infer.config import RunnerConfig
from mini_infer.engine.scheduler import SchedulerOutput
from mini_infer.runner.base import RunnerResult, TimingResult

_MS_PER_S = 1000.0


class SimulatedModelRunner:
    """Models prefill and decode as parallel, budget-limited phases.

    ``prefill_ms = base + per_token * P``
    ``decode_ms  = base + per_token * B + per_context_token * C``

    A step's duration is the slower of the two phases, with ``P`` the prefill
    tokens, ``B`` the decoding requests and ``C`` the total context length those
    requests attend over.
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()

    def execute(
        self, output: SchedulerOutput, *, context_lengths: dict[str, int]
    ) -> RunnerResult:
        return RunnerResult(sampled_tokens={w.request_id: 0 for w in output.decode_work})

    def time_step(
        self, output: SchedulerOutput, *, context_lengths: dict[str, int]
    ) -> TimingResult:
        cfg = self.config
        prefill_ms = cfg.prefill_base_ms + cfg.prefill_ms_per_token * output.num_prefill_tokens
        decode_ms = cfg.decode_base_ms + cfg.decode_ms_per_token * output.num_decode_requests
        decode_ms += cfg.decode_ms_per_context_token * sum(
            context_lengths.get(w.request_id, 0) for w in output.decode_work
        )
        prefill_s = prefill_ms / _MS_PER_S
        decode_s = decode_ms / _MS_PER_S
        return TimingResult(
            duration_s=max(prefill_s, decode_s),
            prefill_cost_s=prefill_s,
            decode_cost_s=decode_s,
        )
