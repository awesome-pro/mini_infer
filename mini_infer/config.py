"""Engine and runner configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Parameters of the modelled execution cost of one forward pass.

    ``prefill_ms = prefill_base_ms + prefill_ms_per_token * P``
    ``decode_ms  = decode_base_ms + decode_ms_per_token * B + decode_ms_per_context_token * C``

    where ``P`` is the number of prefill tokens in the step, ``B`` the number of
    decoding requests and ``C`` their total context length. These are a stated
    performance model, not measurements of any particular GPU.
    """

    prefill_base_ms: float = 2.0
    prefill_ms_per_token: float = 0.08
    decode_base_ms: float = 1.5
    decode_ms_per_token: float = 0.05
    decode_ms_per_context_token: float = 0.004

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) < 0:
                raise ValueError(f"{field.name} must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Scheduling and memory limits of one engine instance."""

    # Model geometry, needed to size KV blocks.
    num_layers: int = 12
    num_kv_heads: int = 4
    head_dim: int = 64
    dtype_bytes: int = 2

    # Compute limits.
    max_batch_tokens: int = 64
    max_running_requests: int = 4
    max_prefill_chunk: int | None = None

    # KV memory limits.
    num_blocks: int = 32
    block_size: int = 16

    # Policy.
    policy: str = "fcfs"
    enable_chunked_prefill: bool = True
    enable_preemption: bool = True
    max_wait_steps: int = 16
    prefill_reservation: int = 16

    runner: RunnerConfig = RunnerConfig()

    def __post_init__(self) -> None:
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.max_prefill_chunk is not None and self.max_prefill_chunk < 1:
            raise ValueError("max_prefill_chunk must be >= 1 when set")
        if self.dtype_bytes < 1:
            raise ValueError("dtype_bytes must be >= 1")

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache consumed by one token of one sequence."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def kv_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def with_(self, **changes: Any) -> EngineConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        payload = dict(data)
        runner = payload.pop("runner", None)
        if isinstance(runner, dict):
            payload["runner"] = RunnerConfig(**runner)
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> EngineConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
