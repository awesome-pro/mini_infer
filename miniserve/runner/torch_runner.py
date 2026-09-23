"""A real transformer runner, with its KV cache in the engine's physical blocks.

This is the runner that makes the runtime more than a simulator. It loads a
HuggingFace causal LM, and for every engine step it runs one forward pass over the
whole batch, reading and writing the *same* physical KV blocks the block manager
allocates. Nothing about the cache is bookkeeping: the tensors attention reads are
the tensors the pool owns.

**Pool layout.** KV lives in ``[num_layers, num_blocks, num_kv_heads, block_size,
head_dim]`` tensors, one for keys and one for values. Logical block ``i`` of a
request covers token positions ``[i * block_size, (i + 1) * block_size)`` and maps to
a physical block through the request's :class:`~miniserve.block_table.BlockTable`,
exactly as the manager sees it.

**One forward per step, not one per request.** All the work in a step is flattened
into a single token sequence: every request contributes its chunk, attention runs
over a gathered copy of each request's cached context, and a block-diagonal causal
mask keeps requests from seeing each other. That is what continuous batching means
once a real model is attached — the batch is the step.

**Attention over paged storage.** For each layer and request, the cached prefix is
gathered from the pool by block table into a contiguous tensor, the freshly computed
chunk is written back into its blocks, and attention runs over prefix plus chunk. The
gather is real; there is no fused paged-attention kernel, and the README says so.

**Time.** :meth:`time_step` reports zero: the work has not happened yet at that point,
and the engine's :class:`~miniserve.clock.WallClock` measures the forward directly.

**Tokens.** A real forward samples the next token while computing the positions it was
granted, so a decoding request always carries exactly one position whose KV is still
pending. The engine books the token on the step that sampled it, which puts the first
token at the end of prefill — where a server first has something to stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from miniserve.config import EngineConfig
from miniserve.engine.request import Request
from miniserve.engine.scheduler import SchedulerOutput
from miniserve.runner.base import RunnerResult, TimingResult

#: A tiny Llama with a real tokenizer: enough to exercise every code path in seconds.
DEFAULT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

#: KV pool element type, derived from the engine's declared KV precision.
#: float64 is here so a check can compare this forward against a reference exactly.
DTYPE_FOR_BYTES = {2: torch.float16, 4: torch.float32, 8: torch.float64}


def engine_config_for_model(model: str, **overrides: Any) -> EngineConfig:
    """An :class:`EngineConfig` whose KV geometry matches ``model``.

    The engine sizes blocks from the config, so the config has to describe the same
    model the runner loaded; deriving it beats asking a person to copy four numbers.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model)
    heads = getattr(hf_config, "num_attention_heads", None)
    kv_heads = getattr(hf_config, "num_key_value_heads", None) or heads
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // heads
    settings: dict[str, Any] = {
        "num_layers": hf_config.num_hidden_layers,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": 2,
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def rotate_half(x: Tensor) -> Tensor:
    """The half-rotation RoPE applies: ``[-x2, x1]``."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class PagedKVCache:
    """Key/value tensors laid out as the engine's block pool."""

    __slots__ = ("block_size", "head_dim", "key", "num_blocks", "num_kv_heads", "value")

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.key = torch.zeros(shape, dtype=dtype, device=device)
        self.value = torch.zeros(shape, dtype=dtype, device=device)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @property
    def num_bytes(self) -> int:
        return 2 * self.key.numel() * self.key.element_size()

    def index_for(self, table, num_tokens: int) -> tuple[Tensor, Tensor]:
        """Physical block and slot for each position in ``[0, num_tokens)``."""
        positions = torch.arange(num_tokens, device=self.key.device)
        blocks = torch.as_tensor(table.physical_blocks, device=self.key.device)
        return blocks[positions // self.block_size], positions % self.block_size

    def write(self, layer: int, table, start: int, keys: Tensor, values: Tensor) -> None:
        """Scatter ``[T, H, D]`` keys and values over the positions ``[start, start + T)``.

        Blocks are written one at a time because a chunk can start mid-block; the loop
        runs over the handful of blocks a chunk touches, not over its tokens.
        """
        num_tokens = int(keys.shape[0])
        size = self.block_size
        for logical in range(start // size, (start + num_tokens - 1) // size + 1):
            low = max(start, logical * size)
            high = min(start + num_tokens, (logical + 1) * size)
            slot = low - logical * size
            span = slice(slot, slot + (high - low))
            chunk_keys = keys[low - start : high - start].transpose(0, 1)
            chunk_values = values[low - start : high - start].transpose(0, 1)
            self.key[layer, table[logical], :, span, :] = chunk_keys
            self.value[layer, table[logical], :, span, :] = chunk_values

    def gather(self, layer: int, blocks: Tensor, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Read a request's cached keys and values as ``[H, T, D]``."""
        key = self.key[layer][blocks, :, slots, :].transpose(0, 1)
        value = self.value[layer][blocks, :, slots, :].transpose(0, 1)
        return key, value


@dataclass(slots=True)
class _Segment:
    """One request's contribution to the flattened batch of a step."""

    request: Request
    start: int
    num_new: int
    end: int
    row_start: int
    tokens: Tensor
    positions: Tensor
    blocks: Tensor
    slots: Tensor

    @property
    def rows(self) -> slice:
        return slice(self.row_start, self.row_start + self.num_new)

    @property
    def last_row(self) -> int:
        return self.row_start + self.num_new - 1


class TorchModelRunner:
    """Executes engine steps with a real model and a real paged KV cache."""

    attends_over_kv = True

    def __init__(
        self,
        config: EngineConfig,
        *,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        self.config = config
        self.device = torch.device(device)
        self.model_name = model
        pool_dtype = dtype or DTYPE_FOR_BYTES.get(config.dtype_bytes)
        if pool_dtype is None:
            raise ValueError(
                f"no KV element type for dtype_bytes={config.dtype_bytes}; "
                f"use one of {sorted(DTYPE_FOR_BYTES)} or pass dtype explicitly"
            )

        hf_config = AutoConfig.from_pretrained(model)
        self.hf_config = hf_config
        self.dtype = pool_dtype
        self.model = (
            AutoModelForCausalLM.from_pretrained(model, dtype=pool_dtype)
            .to(self.device)
            .eval()
        )
        self._validate_geometry(hf_config, pool_dtype)

        #: Vocabulary logits from the last :meth:`execute`, per request. Kept so a
        #: caller can compare this forward against another implementation.
        self.last_logits: dict[str, Tensor] = {}
        self.num_layers = int(hf_config.num_hidden_layers)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = self.config.num_kv_heads
        self.head_dim = self.config.head_dim
        self.scale = self.head_dim**-0.5
        self._validate_structure()
        self._layers = list(self.model.model.layers)
        self.cache = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=pool_dtype,
            device=self.device,
        )

    # ------------------------------------------------------------ description

    @property
    def pool_bytes(self) -> int:
        """Size of the KV pool the engine's block count implies."""
        return self.cache.num_bytes

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "layers": self.num_layers,
            "heads": self.num_heads,
            "kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "blocks": self.config.num_blocks,
            "block_size": self.config.block_size,
            "pool_mib": round(self.pool_bytes / 2**20, 1),
        }

    def _validate_geometry(self, hf_config: Any, pool_dtype: torch.dtype) -> None:
        """The engine sizes KV from the config, so the two must describe one model."""
        observed = {
            "num_layers": (self.config.num_layers, hf_config.num_hidden_layers),
            "num_kv_heads": (
                self.config.num_kv_heads,
                getattr(hf_config, "num_key_value_heads", None)
                or hf_config.num_attention_heads,
            ),
            "head_dim": (
                self.config.head_dim,
                getattr(hf_config, "head_dim", None)
                or hf_config.hidden_size // hf_config.num_attention_heads,
            ),
            "dtype_bytes": (
                self.config.dtype_bytes,
                torch.empty(0, dtype=pool_dtype).element_size(),
            ),
        }
        wrong = [
            f"{name}: config {got} but model {want}"
            for name, (got, want) in observed.items()
            if got != want
        ]
        if wrong:
            raise ValueError(
                "EngineConfig and the loaded model disagree on KV geometry ("
                + "; ".join(wrong)
                + "). Derive the config with engine_config_for_model()."
            )

    def _validate_structure(self) -> None:
        """Check the model has the block layout this forward pass walks."""
        base = self.model.model
        missing = [name for name in ("embed_tokens", "layers", "norm") if not hasattr(base, name)]
        if missing or not list(getattr(base, "layers", [])):
            raise ValueError(
                f"{self.model_name} is not a supported decoder: missing {missing or 'layers'}. "
                "This runner implements the Llama block layout (Llama, Qwen2/3, SmolLM2, "
                "TinyLlama and relatives)."
            )
        layer = base.layers[0]
        for owner, names in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (layer.mlp, ("gate_proj", "up_proj", "down_proj")),
            (layer, ("input_layernorm", "post_attention_layernorm")),
        ):
            absent = [name for name in names if not hasattr(owner, name)]
            if absent:
                raise ValueError(
                    f"{self.model_name} is not a supported decoder: {absent} not found. "
                    "This runner implements the Llama block layout."
                )

    def warmup(self) -> None:
        """One tiny forward, to pay allocator and BLAS setup before the first request.

        It warms the model, not the paged path: the first real step still pays for the
        pool's first touch.
        """
        with torch.no_grad():
            ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            self.model(input_ids=ids)

    # ---------------------------------------------------------------- running

    def time_step(
        self, output: SchedulerOutput, *, context_lengths: dict[str, int]
    ) -> TimingResult:
        """Zero: the forward has not run yet, and the wall clock measures it.

        ``Engine.step`` calls this before :meth:`execute` and then reads the clock, so
        reporting a duration here would count the work twice under a real clock.
        """
        return TimingResult(duration_s=0.0)

    @torch.no_grad()
    def execute(
        self, output: SchedulerOutput, *, context_lengths: dict[str, int]
    ) -> RunnerResult:
        segments = self._segments(output)
        self.last_logits = {}
        if not segments:
            return RunnerResult(sampled_tokens={})

        input_ids = torch.cat([segment.tokens for segment in segments])
        positions = torch.cat([segment.positions for segment in segments])
        hidden = self.model.model.embed_tokens(input_ids)
        cos, sin = self._cos_sin(positions)
        mask = self._attention_mask(segments, sum(s.end for s in segments))
        for index, layer in enumerate(self._layers):
            hidden = self._layer(index, layer, hidden, segments, mask, cos, sin)

        hidden = self.model.model.norm(hidden)
        last_rows = torch.tensor(
            [segment.last_row for segment in segments], device=self.device
        )
        logits = self.model.lm_head(hidden[last_rows])
        sampled = logits.argmax(dim=-1)
        self.last_logits = {
            segment.request.id: logits[index].detach()
            for index, segment in enumerate(segments)
        }
        return RunnerResult(
            sampled_tokens={
                segment.request.id: int(sampled[index])
                for index, segment in enumerate(segments)
            }
        )

    def _segments(self, output: SchedulerOutput) -> list[_Segment]:
        """Slice each scheduled request's share of the sequence into a flat batch."""
        segments: list[_Segment] = []
        row = 0
        for work in output.work:
            if work.num_new_tokens <= 0:
                continue
            request = work.request
            start = request.num_computed_tokens
            end = start + work.num_new_tokens
            sequence = request.prompt_tokens + request.generated_tokens
            if end > len(sequence):
                raise ValueError(
                    f"{request.id}: step needs positions [{start}, {end}) but the "
                    f"sequence holds {len(sequence)} tokens. A runner that samples every "
                    f"step keeps exactly one position pending; this request has "
                    f"{request.num_uncomputed_tokens}."
                )
            blocks, slots = self.cache.index_for(request.block_table, end)
            segments.append(
                _Segment(
                    request=request,
                    start=start,
                    num_new=work.num_new_tokens,
                    end=end,
                    row_start=row,
                    tokens=torch.tensor(
                        sequence[start:end], dtype=torch.long, device=self.device
                    ),
                    positions=torch.arange(start, end, device=self.device),
                    blocks=blocks,
                    slots=slots,
                )
            )
            row += work.num_new_tokens
        return segments

    def _cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """RoPE tables for absolute positions, shaped to broadcast over heads.

        The tables are built in the model's dtype. Taking that dtype from the input
        ids instead would silently truncate every cosine to an integer, which is
        exactly the kind of bug that still produces plausible-looking tokens.
        """
        rotary = getattr(self.model.model, "rotary_emb", None)
        if rotary is not None:
            hidden = torch.zeros(
                (1, positions.numel(), self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            cos, sin = rotary(hidden, positions.unsqueeze(0))
            return cos.squeeze(0).unsqueeze(1), sin.squeeze(0).unsqueeze(1)
        return self._manual_cos_sin(positions)

    def _manual_cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Fallback RoPE tables, for a transformers version without ``rotary_emb``."""
        theta = 10000.0
        params = getattr(self.hf_config, "rope_parameters", None) or {}
        if isinstance(params, dict):
            theta = float(params.get("rope_theta", theta))
        inverse = 1.0 / (
            theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
                / self.head_dim
            )
        )
        angles = torch.outer(positions.to(torch.float32), inverse)
        angles = torch.cat((angles, angles), dim=-1)
        return (
            angles.cos().to(self.dtype).unsqueeze(1),
            angles.sin().to(self.dtype).unsqueeze(1),
        )

    def _attention_mask(self, segments: list[_Segment], total_keys: int) -> Tensor:
        """Block-diagonal causal mask: one request never attends to another's KV."""
        mask = torch.full(
            (sum(s.num_new for s in segments), total_keys),
            float("-inf"),
            dtype=self.dtype,
            device=self.device,
        )
        column = 0
        for segment in segments:
            allowed = segment.positions.unsqueeze(1) >= torch.arange(
                segment.end, device=self.device
            ).unsqueeze(0)
            mask[segment.rows, column : column + segment.end] = torch.where(
                allowed, 0.0, float("-inf")
            ).to(self.dtype)
            column += segment.end
        return mask

    def _layer(
        self,
        index: int,
        layer: Any,
        hidden: Tensor,
        segments: list[_Segment],
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """One decoder block: attention over the paged cache, then the MLP."""
        attention = layer.self_attn
        residual = hidden
        normed = layer.input_layernorm(hidden)
        query = attention.q_proj(normed).view(-1, self.num_heads, self.head_dim)
        key = attention.k_proj(normed).view(-1, self.num_kv_heads, self.head_dim)
        value = attention.v_proj(normed).view(-1, self.num_kv_heads, self.head_dim)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin

        attended = self._attend(index, query, key, value, segments, mask)
        hidden = residual + attention.o_proj(attended)

        residual = hidden
        normed = layer.post_attention_layernorm(hidden)
        mlp = layer.mlp
        hidden = residual + mlp.down_proj(mlp.act_fn(mlp.gate_proj(normed)) * mlp.up_proj(normed))
        return hidden

    def _attend(
        self,
        index: int,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        segments: list[_Segment],
        mask: Tensor,
    ) -> Tensor:
        """Write this chunk into its blocks, gather each context, attend, merge heads."""
        for segment in segments:
            self.cache.write(
                index, segment.request.block_table, segment.start,
                key[segment.rows], value[segment.rows],
            )

        gathered = [self.cache.gather(index, s.blocks, s.slots) for s in segments]
        keys = torch.cat([pair[0] for pair in gathered], dim=1)
        values = torch.cat([pair[1] for pair in gathered], dim=1)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            keys = keys.repeat_interleave(repeats, dim=0)
            values = values.repeat_interleave(repeats, dim=0)

        scores = torch.matmul(query.transpose(0, 1), keys.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores + mask, dim=-1)
        merged = torch.matmul(weights, values).transpose(0, 1)
        return merged.reshape(query.shape[0], self.num_heads * self.head_dim)
