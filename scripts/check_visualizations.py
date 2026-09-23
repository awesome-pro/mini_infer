"""Checks for the visualization layer: shaping, rendering, and the README's figures.

The shaping functions are checked without plotting installed, because that is the
property that keeps matplotlib optional. The renderers are checked only when
matplotlib is importable; when it is not, those checks pass with a printed note rather
than failing a machine that deliberately did not install the ``plot`` extra.

Run with::

    python scripts/check_visualizations.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _check import check, expect_raises, report

from miniserve import Engine, EngineConfig, Request, RunnerConfig
from miniserve.visualizations.plots import (
    HistogramPanel,
    Panel,
    Point,
    histogram,
    plot_bars,
    plot_histograms,
    plot_lines,
    plot_scatter,
    plot_step_series,
    plot_timeline,
    series_from_rows,
    step_series,
    timeline_grid,
)

ROOT = Path(__file__).resolve().parent.parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ZERO_COST = RunnerConfig(
    prefill_base_ms=0.0,
    prefill_ms_per_token=1.0,
    decode_base_ms=0.0,
    decode_ms_per_token=1.0,
    decode_ms_per_context_token=0.0,
)

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    HAS_MATPLOTLIB = False


def small_run(*, policy: str = "fcfs", prompts: tuple[int, ...] = (12, 8)) -> Engine:
    engine = Engine(
        EngineConfig(
            policy=policy,
            max_batch_tokens=16,
            max_running_requests=4,
            num_blocks=32,
            block_size=8,
            runner=ZERO_COST,
        )
    )
    for index, length in enumerate(prompts):
        engine.submit(
            Request(
                prompt_tokens=list(range(length)),
                max_new_tokens=4,
                arrival_time=0.0,
                id=f"r{index}",
            )
        )
    engine.run_to_completion()
    return engine


# ------------------------------------------------------------------- shaping


def check_timeline_grid_is_one_row_per_request() -> None:
    engine = small_run(prompts=(12, 8, 20))
    requests = list(engine.requests.values())
    grid = timeline_grid(engine.history, requests)

    assert grid.num_requests == 3
    assert grid.num_steps == len(engine.history)
    assert grid.request_ids == ("r0", "r1", "r2")
    assert grid.step_indices == tuple(step.index for step in engine.history)
    assert grid.num_active_cells > 0
    for row in grid.tokens:
        assert len(row) == grid.num_steps
    for row in grid.kinds:
        assert len(row) == grid.num_steps


def check_timeline_grid_counts_work_not_slots() -> None:
    """Every prompt and output token appears exactly once (no preemption here)."""
    engine = small_run(prompts=(12, 8, 20))
    requests = list(engine.requests.values())
    grid = timeline_grid(engine.history, requests)

    for row, request in enumerate(requests):
        prefill = sum(
            grid.tokens[row][column]
            for column in range(grid.num_steps)
            if grid.kinds[row][column] == "prefill"
        )
        decode = [
            grid.tokens[row][column]
            for column in range(grid.num_steps)
            if grid.kinds[row][column] == "decode"
        ]
        assert prefill == request.prompt_len, f"{request.id}: {prefill} prefill tokens drawn"
        assert all(tokens == 1 for tokens in decode), "a decode cell is one token"
        assert len(decode) == request.num_generated
        assert grid.tokens_for(request.id, request.admitted_step) >= 0


def check_timeline_grid_ignores_requests_outside_the_run() -> None:
    engine = small_run(prompts=(8,))
    requests = list(engine.requests.values())
    outsider = Request(prompt_tokens=list(range(4)), max_new_tokens=1, arrival_time=0.0, id="ghost")

    grid = timeline_grid(engine.history, [*requests, outsider])

    assert grid.request_ids == ("r0", "ghost")
    assert sum(grid.tokens[1]) == 0, "a request that never ran must not appear as work"
    assert all(kind == "" for kind in grid.kinds[1])


def check_step_series_matches_the_run() -> None:
    engine = small_run(prompts=(12, 8))
    series = step_series(engine)

    assert series.num_steps == len(engine.history)
    assert series.times_ms == tuple(step.end_time * 1e3 for step in engine.history)
    assert all(later >= earlier for earlier, later in pairwise(series.times_ms))
    assert all(0.0 <= value <= 1.0 for value in series.utilization)
    assert all(0.0 <= value < 1.0 for value in series.fragmentation)
    assert any(value > 0 for value in series.fragmentation), "8-token blocks must waste a tail"
    assert all(value >= 0 for value in series.running)
    assert all(value >= 0 for value in series.waiting)
    assert any(value > 0 for value in series.prefill_tokens)
    assert any(value > 0 for value in series.decode_requests)


def check_step_series_survives_a_collector_without_steps() -> None:
    """A swapped collector must not misalign the traces."""
    engine = small_run(prompts=(12,))
    engine.metrics.steps.clear()

    series = step_series(engine)

    assert series.num_steps == len(engine.history)
    assert series.fragmentation == (0.0,) * series.num_steps


def check_series_from_rows_skips_non_numeric() -> None:
    rows = [
        {"label": "a", "p95_ttft_ms": 12.5, "stalled": False},
        {"label": "b", "p95_ttft_ms": 4, "stalled": True},
        {"label": "c", "stalled": False},
    ]
    assert series_from_rows(rows, "p95_ttft_ms") == (12.5, 4.0)
    assert series_from_rows(rows, "stalled") == ()
    assert series_from_rows(rows, "missing") == ()


def check_histogram_counts_every_sample() -> None:
    values = [1.0, 2.0, 2.5, 3.0, 10.0, 10.0]
    bins = histogram(values, bins=9)

    assert len(bins.counts) == 9
    assert len(bins.edges) == 10
    assert bins.total == len(values)
    assert all(later > earlier for earlier, later in pairwise(bins.edges))
    assert len(bins.centers) == 9


def check_histogram_handles_degenerate_input() -> None:
    assert histogram([], bins=4) == histogram([], bins=4)
    empty = histogram([], bins=4)
    assert empty.counts == () and empty.edges == () and empty.total == 0

    single = histogram([5.0, 5.0, 5.0], bins=4)
    assert single.total == 3
    assert sum(single.counts) == 3

    with_nan = histogram([1.0, float("nan"), 2.0], bins=2)
    assert with_nan.total == 2, "NaN samples are dropped, not binned"


def check_histogram_rejects_zero_bins() -> None:
    expect_raises(ValueError, "bins", lambda: histogram([1.0], bins=0))


def check_shaping_does_not_import_matplotlib() -> None:
    """The extra stays optional: shaping must work with no plotting library loaded."""
    script = (
        "import sys;"
        f"sys.path.insert(0, {str(ROOT)!r});"
        "from miniserve.visualizations.plots import histogram, series_from_rows;"
        "histogram([1.0, 2.0]);"
        "series_from_rows([{'a': 1}], 'a');"
        "print('matplotlib' in sys.modules)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "False", (
        f"importing the shaping helpers pulled in matplotlib: {finished.stdout!r}"
    )


# ------------------------------------------------------------------ rendering


def _is_png(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000 and path.read_bytes()[:8] == PNG_MAGIC


def _require_matplotlib(what: str) -> bool:
    if HAS_MATPLOTLIB:
        return True
    print(f"  note: skipping {what}; matplotlib is not installed")
    return False


def check_engine_figures_render() -> None:
    if not _require_matplotlib("figure rendering"):
        return
    engine = small_run(prompts=(12, 8, 20))
    requests = list(engine.requests.values())

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "nested"  # exercises directory creation
        timeline = plot_timeline(
            timeline_grid(engine.history, requests), out / "timeline.png", title="test"
        )
        series = plot_step_series(step_series(engine), out / "series.png")
        assert _is_png(timeline), f"{timeline} is not a PNG"
        assert _is_png(series), f"{series} is not a PNG"


def check_result_figures_render() -> None:
    if not _require_matplotlib("figure rendering"):
        return
    rows = [
        {"output_tok_per_s": 100.0, "p95_ttft_ms": 50.0},
        {"output_tok_per_s": 250.0, "p95_ttft_ms": 120.0},
        {"output_tok_per_s": 300.0, "p95_ttft_ms": 400.0},
    ]
    x = [1, 2, 4]

    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        lines = plot_lines(
            x,
            [Panel("throughput", "tokens/s", {"out": series_from_rows(rows, "output_tok_per_s")})],
            out / "lines.png",
            xlabel="rate",
        )
        bars = plot_bars(
            ["a", "b"],
            [Panel("ttft", "ms", {"p95": [10.0, 20.0]})],
            out / "bars.png",
        )
        assert _is_png(lines), f"{lines} is not a PNG"
        assert _is_png(bars), f"{bars} is not a PNG"


def check_distribution_and_scatter_figures_render() -> None:
    if not _require_matplotlib("figure rendering"):
        return
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        histograms = plot_histograms(
            [
                HistogramPanel("ttft", "ms", [1.0, 2.0, 3.0, 4.0, 5.0], reference=4.0),
                HistogramPanel("itl", "ms", []),
            ],
            out / "hist.png",
        )
        scatter = plot_scatter(
            [Point("a", 1.0, 2.0), Point("b", 2.0, 1.0)],
            out / "scatter.png",
            xlabel="x",
            ylabel="y",
        )
        assert _is_png(histograms), f"{histograms} is not a PNG"
        assert _is_png(scatter), f"{scatter} is not a PNG"


def check_empty_inputs_are_rejected_loudly() -> None:
    if not _require_matplotlib("figure rendering"):
        return
    expect_raises(ValueError, "nothing to plot", lambda: plot_lines([], [], Path("/tmp/x.png")))
    expect_raises(
        ValueError,
        "nothing to plot",
        lambda: plot_scatter([], Path("/tmp/x.png"), xlabel="", ylabel=""),
    )
    expect_raises(
        ValueError, "nothing to plot", lambda: plot_bars(["a"], [], Path("/tmp/x.png"))
    )
    expect_raises(
        ValueError, "nothing to plot", lambda: plot_histograms([], Path("/tmp/x.png"))
    )


# ------------------------------------------------------------------- README


def check_readme_figures_exist() -> None:
    """Every figure the README shows must be a real PNG in the repository."""
    readme = ROOT / "README.md"
    assert readme.exists(), "the README is the deliverable for this phase"
    text = readme.read_text()
    references = sorted(
        {
            part.split(")")[0].split('"')[0].strip()
            for part in text.split("](")[1:]
            if part.startswith("figures/")
        }
    )
    assert references, "the README should show the measured experiments"
    for reference in references:
        path = ROOT / reference
        assert _is_png(path), f"{reference} is referenced by the README but is not a PNG file"


def main() -> int:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed: rendering checks will be skipped")
    check("timeline grid is one row per request", check_timeline_grid_is_one_row_per_request)
    check("timeline grid counts work, not slots", check_timeline_grid_counts_work_not_slots)
    check(
        "timeline grid ignores requests outside the run",
        check_timeline_grid_ignores_requests_outside_the_run,
    )
    check("step series matches the run", check_step_series_matches_the_run)
    check(
        "step series survives a collector without steps",
        check_step_series_survives_a_collector_without_steps,
    )
    check("series_from_rows skips non-numeric", check_series_from_rows_skips_non_numeric)
    check("histogram counts every sample", check_histogram_counts_every_sample)
    check("histogram handles degenerate input", check_histogram_handles_degenerate_input)
    check("histogram rejects zero bins", check_histogram_rejects_zero_bins)
    check("shaping does not import matplotlib", check_shaping_does_not_import_matplotlib)
    check("engine figures render", check_engine_figures_render)
    check("result figures render", check_result_figures_render)
    check("distribution and scatter figures render", check_distribution_and_scatter_figures_render)
    check("empty inputs are rejected loudly", check_empty_inputs_are_rejected_loudly)
    check("README figures exist", check_readme_figures_exist)
    return report("phase 7 visualizations")


if __name__ == "__main__":
    raise SystemExit(main())
