"""Checks for the real-model runner: paged attention, tokens and engine invariants.

The model is a tiny randomly initialised Llama built in-process and saved to a
temporary directory, so these checks need no network and no download. Correctness is
pinned against HuggingFace's own forward pass rather than against a stored expectation:
a weak implementation could easily produce plausible-looking text, but it cannot
reproduce ``generate``'s greedy token sequence over a paged cache it built itself.

Run with::

    python scripts/check_torch_runner.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, close, expect_raises, report

from miniserve import Engine, EngineConfig, Request, VirtualClock, WallClock
from miniserve.engine.engine import StepEventKind
from miniserve.engine.request import RequestStatus

ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    HAS_TORCH = False


# --------------------------------------------------------------- tiny fixture

_FIXTURE: dict[str, object] = {}
_KEEPALIVE: list[tempfile.TemporaryDirectory] = []


def tiny_model() -> tuple[object, Path]:
    """A deterministic two-layer Llama on disk, built once per process."""
    if "path" in _FIXTURE:
        return _FIXTURE["model"], _FIXTURE["path"]  # type: ignore[return-value]

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    directory = tempfile.TemporaryDirectory()
    _KEEPALIVE.append(directory)
    path = Path(directory.name)
    model.save_pretrained(path)

    _FIXTURE["model"] = model
    _FIXTURE["path"] = path
    _FIXTURE["config"] = config
    return model, path


def engine_config(**overrides) -> EngineConfig:
    settings: dict = {
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype_bytes": 4,
        "num_blocks": 64,
        "block_size": 8,
        "max_batch_tokens": 64,
        "max_running_requests": 4,
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def runner_for(config: EngineConfig):
    from miniserve.runner.torch_runner import TorchModelRunner

    _, path = tiny_model()
    return TorchModelRunner(config, model=str(path))


def run_prompts(
    prompts: list[list[int]], *, max_new: int, clock=None, **overrides
) -> tuple[object, object]:
    config = engine_config(**overrides)
    engine = Engine(config, runner=runner_for(config), clock=clock or VirtualClock())
    for index, prompt in enumerate(prompts):
        engine.submit(
            Request(
                prompt_tokens=prompt,
                max_new_tokens=max_new,
                arrival_time=0.0,
                id=f"r{index}",
            )
        )
    engine.run_to_completion(max_steps=20_000)
    return engine, config


def greedy_reference(prompt: list[int], max_new: int) -> list[int]:
    model, _ = tiny_model()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([prompt]),
            do_sample=False,
            max_new_tokens=max_new,
            min_new_tokens=max_new,
        )
    return out[0, len(prompt) :].tolist()


PROMPT_A = [3, 11, 7, 42, 5, 9, 21]
PROMPT_B = [17, 4, 29, 1, 33]


# ------------------------------------------------------------------ packaging


def check_importing_the_package_does_not_import_torch() -> None:
    """Torch is an extra: the runtime must import without it."""
    script = (
        "import sys;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        "import miniserve;"
        "miniserve.Engine(miniserve.EngineConfig()).step();"
        "print('torch' in sys.modules)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "False", (
        f"importing miniserve pulled in torch: {finished.stdout!r}"
    )


def check_config_can_be_derived_from_the_model() -> None:
    from miniserve.runner.torch_runner import TorchModelRunner, engine_config_for_model

    _, path = tiny_model()
    derived = engine_config_for_model(str(path))
    assert derived.num_layers == 2
    assert derived.num_kv_heads == 2
    assert derived.head_dim == 8
    assert derived.dtype_bytes == 2
    # The derived config is by definition consistent with the model it describes.
    TorchModelRunner(derived, model=str(path))


def check_geometry_mismatch_is_rejected() -> None:
    from miniserve.runner.torch_runner import TorchModelRunner

    _, path = tiny_model()
    wrong = engine_config(num_layers=99, num_kv_heads=7)
    expect_raises(
        ValueError, "disagree on KV geometry", lambda: TorchModelRunner(wrong, model=str(path))
    )
    for expected in ("num_layers: config 99 but model 2", "num_kv_heads: config 7 but model 2"):
        expect_raises(ValueError, expected, lambda: TorchModelRunner(wrong, model=str(path)))


def check_pool_matches_the_declared_block_count() -> None:
    from miniserve.runner.torch_runner import TorchModelRunner, engine_config_for_model

    _, path = tiny_model()
    config = engine_config_for_model(str(path), num_blocks=32, block_size=8)
    runner = TorchModelRunner(config, model=str(path))

    expected = (
        2  # keys and values
        * 2  # layers
        * 32  # blocks
        * 2  # kv heads
        * 8  # tokens per block
        * 8  # head dim
        * 2  # bytes per element (dtype_bytes=2 -> float16)
    )
    assert runner.pool_bytes == expected, f"{runner.pool_bytes} != {expected}"
    described = runner.describe()
    assert described["blocks"] == 32
    assert described["block_size"] == 8
    assert described["dtype"] in {"float16", "float32"}


def check_pool_dtype_follows_the_config() -> None:
    from miniserve.runner.torch_runner import TorchModelRunner, engine_config_for_model

    _, path = tiny_model()
    half = TorchModelRunner(engine_config(dtype_bytes=2), model=str(path))
    assert half.dtype == torch.float16
    single = engine_config_for_model(str(path), dtype_bytes=4)
    assert TorchModelRunner(single, model=str(path)).dtype == torch.float32
    expect_raises(
        ValueError,
        "no KV element type",
        lambda: TorchModelRunner(engine_config(dtype_bytes=1), model=str(path)),
    )


# ------------------------------------------------------------------ correctness


def check_sampled_logits_match_huggingface() -> None:
    """The paged forward must agree with the reference forward, numerically."""
    config = engine_config()
    engine = Engine(config, runner=runner_for(config))
    engine.submit(Request(prompt_tokens=PROMPT_A, max_new_tokens=1, arrival_time=0.0, id="r0"))
    engine.step()

    model, _ = tiny_model()
    with torch.no_grad():
        reference = model(input_ids=torch.tensor([PROMPT_A])).logits[0, -1]
    mine = engine.runner.last_logits["r0"]
    assert mine.shape == reference.shape
    worst = float((mine - reference).abs().max())
    # Not a smoke test: a wrong mask, a misplaced rotation or a truncated dtype all
    # still produce plausible tokens, and all show up here as a difference in logits.
    assert worst < 1e-5, f"logits differ by {worst:.3e} from the reference forward"


def check_float64_reproduces_the_reference_exactly() -> None:
    """In float64 the two implementations should agree to machine precision.

    This is the strongest statement available: it rules out a mathematically
    different computation, leaving only float32 accumulation order.
    """
    from miniserve.runner.torch_runner import TorchModelRunner, engine_config_for_model

    _, path = tiny_model()
    config = engine_config_for_model(str(path), dtype_bytes=8)
    runner = TorchModelRunner(config, model=str(path), dtype=torch.float64)
    engine = Engine(config, runner=runner)
    engine.submit(Request(prompt_tokens=PROMPT_A, max_new_tokens=1, arrival_time=0.0, id="r0"))
    engine.step()

    model, _ = tiny_model()
    reference = model.to(torch.float64)
    with torch.no_grad():
        want = reference(input_ids=torch.tensor([PROMPT_A])).logits[0, -1]
    worst = float((runner.last_logits["r0"] - want).abs().max())
    assert worst < 1e-12, f"float64 forward differs by {worst:.3e} from the reference"


def check_greedy_tokens_match_generate() -> None:
    engine, _ = run_prompts([PROMPT_A], max_new=8)
    got = engine.requests["r0"].generated_tokens
    assert got == greedy_reference(PROMPT_A, 8), f"{got} != {greedy_reference(PROMPT_A, 8)}"


def check_chunked_prefill_matches_one_shot() -> None:
    """Splitting prefill across steps must not change a single token."""
    want = greedy_reference(PROMPT_A, 6)
    for overrides in (
        {"max_batch_tokens": 4, "max_prefill_chunk": 4},
        {"max_batch_tokens": 8, "block_size": 1, "num_blocks": 256},
        {"max_batch_tokens": 3, "block_size": 2, "num_blocks": 128},
    ):
        engine, _ = run_prompts([PROMPT_A], max_new=6, **overrides)
        got = engine.requests["r0"].generated_tokens
        assert got == want, f"{overrides}: {got} != {want}"


def check_batching_does_not_change_tokens() -> None:
    """Requests sharing a step must not read each other's KV."""
    alone = [greedy_reference(PROMPT_A, 5), greedy_reference(PROMPT_B, 5)]
    engine, _ = run_prompts([PROMPT_A, PROMPT_B], max_new=5, max_batch_tokens=64)
    batched = [engine.requests[f"r{i}"].generated_tokens for i in range(2)]

    assert batched == alone, f"batched {batched} != isolated {alone}"


def check_preemption_preserves_tokens() -> None:
    """Eviction and recomputation must be invisible in the output."""
    want = [greedy_reference(PROMPT_A, 5), greedy_reference(PROMPT_B, 5)]
    engine, _ = run_prompts(
        [PROMPT_A, PROMPT_B],
        max_new=5,
        num_blocks=8,
        block_size=2,
        max_batch_tokens=8,
        max_running_requests=2,
    )
    got = [engine.requests[f"r{i}"].generated_tokens for i in range(2)]
    evictions = sum(request.num_preemptions for request in engine.requests.values())

    assert evictions > 0, "this pool must force recomputation"
    assert got == want, f"recomputation changed the output: {got} != {want}"
    assert len(engine.finished) == 2


def check_a_shared_prefix_generates_the_same_tokens() -> None:
    """The sharpest test of prefix caching: a request that never computed its own prompt.

    The second request reads another request's keys and values, gathered from the same
    physical blocks. If the sharing were even slightly wrong — wrong block, wrong
    position, wrong rotation — the logits would move and the greedy tokens would differ.
    """
    from miniserve.runner.torch_runner import TorchModelRunner, engine_config_for_model

    _, path = tiny_model()
    want = greedy_reference(PROMPT_A, 4)

    config = engine_config_for_model(
        str(path), enable_prefix_cache=True, block_size=4, num_blocks=64, max_batch_tokens=64
    )
    runner = TorchModelRunner(config, model=str(path))
    engine = Engine(config, runner=runner)

    first = engine.submit(
        Request(prompt_tokens=PROMPT_A, max_new_tokens=4, arrival_time=0.0, id="first")
    )
    engine.run_to_completion()
    memory = engine.memory
    assert memory is not None
    assert memory.num_cached_blocks > 0, "the first request must leave its blocks cached"

    later = engine.submit(
        Request(prompt_tokens=PROMPT_A, max_new_tokens=4, arrival_time=0.0, id="later")
    )
    engine.run_to_completion()

    metrics = engine.metrics.summary(elapsed=engine.clock.now())
    assert metrics.cached_prefix_tokens > 0, "the second request must actually reuse KV"
    assert later.generated_tokens == want, (
        f"shared KV changed the output: {later.generated_tokens} != {want}"
    )
    assert first.generated_tokens == want


def check_kv_invariants_hold_through_a_real_run() -> None:
    config = engine_config(num_blocks=8, block_size=4, max_batch_tokens=8, max_running_requests=2)
    engine = Engine(config, runner=runner_for(config))
    for index, prompt in enumerate((PROMPT_A, PROMPT_B)):
        engine.submit(
            Request(prompt_tokens=prompt, max_new_tokens=5, arrival_time=0.0, id=f"r{index}")
        )

    for step in engine.run():
        memory = engine.memory
        assert memory is not None
        memory.assert_invariants()
        assert step.output.num_scheduled_tokens <= config.max_batch_tokens

    memory = engine.memory
    assert memory is not None
    assert memory.num_free_blocks() == memory.num_blocks, "all KV must come back"
    for request in engine.requests.values():
        assert request.status is RequestStatus.FINISHED
        assert request.num_generated == request.max_new_tokens
        # The last token sampled has no KV: there is no next step to compute it in,
        # and nothing will attend to it. One pending position is the convention.
        assert request.num_computed_tokens == request.num_tokens - 1
        assert request.block_table.num_blocks == 0


# ------------------------------------------------------------------- contract


def check_one_position_is_pending_while_decoding() -> None:
    """The engine's decode phase exists to compute exactly one pending position."""
    config = engine_config()
    engine = Engine(config, runner=runner_for(config))
    request = engine.submit(
        Request(prompt_tokens=PROMPT_A, max_new_tokens=4, arrival_time=0.0, id="r0")
    )

    for step in engine.run():
        if step.index == 0:
            assert request.num_computed_tokens == len(PROMPT_A)
            assert request.num_tokens == len(PROMPT_A) + 1, "prefill books the first token"
        if request.status is RequestStatus.DECODING:
            assert request.num_uncomputed_tokens == 1, (
                f"step {step.index}: {request.num_uncomputed_tokens} positions pending"
            )


def check_first_token_lands_at_the_end_of_prefill() -> None:
    config = engine_config()
    engine = Engine(config, runner=runner_for(config), clock=WallClock())
    request = engine.submit(
        Request(prompt_tokens=PROMPT_A, max_new_tokens=4, arrival_time=0.0, id="r0")
    )

    steps = engine.run_to_completion()

    first = steps[0]
    assert first.events_of(StepEventKind.DECODED), "the prefill step streams the first token"
    assert first.num_prefill_tokens == len(PROMPT_A)
    assert request.first_token_time is not None
    assert request.first_token_time <= first.end_time, "TTFT belongs to the prefill step"
    # One forward produced the first token, so the run needs max_new - 1 decodes.
    assert len(steps) == 1 + (4 - 1), f"{len(steps)} steps for 4 tokens"
    assert all(step.duration > 0 for step in steps), "the wall clock measures every step"


def check_time_step_reports_zero() -> None:
    """The forward has not run when the engine asks for a duration."""
    config = engine_config()
    runner = runner_for(config)
    engine = Engine(config, runner=runner)
    engine.submit(Request(prompt_tokens=PROMPT_A, max_new_tokens=1, arrival_time=0.0, id="r0"))
    engine.step()

    timing = runner.time_step(engine.history[-1].output, context_lengths={})
    close(timing.duration_s, 0.0, abs_=1e-12)


def main() -> int:
    check(
        "importing the package does not import torch",
        check_importing_the_package_does_not_import_torch,
    )
    if not HAS_TORCH:
        print("torch is not installed: skipping the runner checks (pip install -e '.[torch]')")
        return report("phase 8 real model runner")

    check("config can be derived from the model", check_config_can_be_derived_from_the_model)
    check("geometry mismatch is rejected", check_geometry_mismatch_is_rejected)
    check("pool matches the declared block count", check_pool_matches_the_declared_block_count)
    check("pool dtype follows the config", check_pool_dtype_follows_the_config)
    check("sampled logits match HuggingFace", check_sampled_logits_match_huggingface)
    check(
        "float64 reproduces the reference exactly",
        check_float64_reproduces_the_reference_exactly,
    )
    check("greedy tokens match generate()", check_greedy_tokens_match_generate)
    check("chunked prefill matches one shot", check_chunked_prefill_matches_one_shot)
    check("batching does not change tokens", check_batching_does_not_change_tokens)
    check("preemption preserves tokens", check_preemption_preserves_tokens)
    check(
        "a shared prefix generates the same tokens",
        check_a_shared_prefix_generates_the_same_tokens,
    )
    check("KV invariants hold through a real run", check_kv_invariants_hold_through_a_real_run)
    check("one position pending while decoding", check_one_position_is_pending_while_decoding)
    check(
        "first token lands at the end of prefill",
        check_first_token_lands_at_the_end_of_prefill,
    )
    check("time_step reports zero", check_time_step_reports_zero)
    return report("phase 8 real model runner")


if __name__ == "__main__":
    raise SystemExit(main())
