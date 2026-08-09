"""Runner package: simulated and torch-backed execution backends."""

from mini_infer.runner.base import RunnerResult, TimingResult
from mini_infer.runner.simulated_runner import SimulatedModelRunner

__all__ = ["RunnerResult", "SimulatedModelRunner", "TimingResult"]
