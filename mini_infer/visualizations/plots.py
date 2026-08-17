"""Figures for documents.

Text is the primary renderer — see :mod:`mini_infer.visualizations.charts` and
:mod:`mini_infer.visualizations.timeline`. This module exists for the README and for
the project write-up, where a picture carries an argument that a table cannot, and it
is the only place in the package that wants matplotlib.

Data shaping and drawing are deliberately separate. :func:`timeline_grid`,
:func:`step_series`, :func:`histogram` and :func:`series_from_rows` are ordinary
functions over engine history, and the ``Panel`` dataclasses are plain values: all of
them can be built and checked with no plotting library installed. Only the ``plot_*``
functions import matplotlib, so the extra stays optional and a missing install fails
with one clear sentence instead of an ImportError at package import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mini_infer.engine.engine import Engine, EngineStep
from mini_infer.engine.request import Request

#: Idle / prefill / decode, in the order the timeline colormap expects.
KIND_CODE = {"": 0, "prefill": 1, "decode": 2}
KIND_COLORS = ("#e9ecef", "#4c72b0", "#dd8452")
PALETTE = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3")
#: Series line styles, so two overlapping curves stay distinguishable.
LINESTYLES = ("-", "--", ":", "-.")
NO_DATA = "(no data)"


# ------------------------------------------------------------------- shaping


@dataclass(frozen=True, slots=True)
class TimelineGrid:
    """Which request ran in which step, and how much work it was granted."""

    request_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    #: ``tokens[row][column]`` for request ``row`` in step ``column``.
    tokens: tuple[tuple[int, ...], ...]
    #: ``kinds[row][column]`` is ``"prefill"``, ``"decode"`` or ``""``.
    kinds: tuple[tuple[str, ...], ...]

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_steps(self) -> int:
        return len(self.step_indices)

    @property
    def num_active_cells(self) -> int:
        return sum(1 for row in self.kinds for kind in row if kind)

    def tokens_for(self, request_id: str, step_index: int) -> int:
        row = self.request_ids.index(request_id)
        column = self.step_indices.index(step_index)
        return self.tokens[row][column]


def timeline_grid(
    steps: Sequence[EngineStep], requests: Sequence[Request]
) -> TimelineGrid:
    """Project a run onto a request x step grid.

    Only tokens that were actually granted appear: a zero-token grant claims a slot
    without doing work, so drawing it would overstate the schedule.
    """
    ids = tuple(request.id for request in requests)
    row_of = {request_id: row for row, request_id in enumerate(ids)}
    columns = tuple(steps)
    tokens = [[0] * len(columns) for _ in ids]
    kinds = [[""] * len(columns) for _ in ids]

    for column, step in enumerate(columns):
        for work in step.output.work:
            row = row_of.get(work.request_id)
            if row is None or work.num_new_tokens <= 0:
                continue
            tokens[row][column] += work.num_new_tokens
            kinds[row][column] = work.kind.value

    return TimelineGrid(
        request_ids=ids,
        step_indices=tuple(step.index for step in columns),
        tokens=tuple(tuple(row) for row in tokens),
        kinds=tuple(tuple(row) for row in kinds),
    )


@dataclass(frozen=True, slots=True)
class StepSeries:
    """Per-step traces of one run, ready to plot against time."""

    times_ms: tuple[float, ...]
    utilization: tuple[float, ...]
    fragmentation: tuple[float, ...]
    running: tuple[int, ...]
    waiting: tuple[int, ...]
    prefill_tokens: tuple[int, ...]
    decode_requests: tuple[int, ...]
    preemptions: tuple[int, ...]

    @property
    def num_steps(self) -> int:
        return len(self.times_ms)


def step_series(engine: Engine) -> StepSeries:
    """Collect the step traces of a finished run.

    Fragmentation is sampled by the metrics collector rather than the engine, so it
    is merged in here; if the collector was swapped for one that did not run, the
    trace falls back to zeros rather than misaligning with the steps.
    """
    history = engine.history
    recorded = engine.metrics.steps
    fragmentation = (
        [step.internal_fragmentation for step in recorded]
        if len(recorded) == len(history)
        else [0.0] * len(history)
    )
    return StepSeries(
        times_ms=tuple(step.end_time * 1e3 for step in history),
        utilization=tuple(step.kv_utilization for step in history),
        fragmentation=tuple(fragmentation),
        running=tuple(step.running_requests for step in history),
        waiting=tuple(step.waiting_requests for step in history),
        prefill_tokens=tuple(step.num_prefill_tokens for step in history),
        decode_requests=tuple(step.num_decode_requests for step in history),
        preemptions=tuple(step.num_preemptions for step in history),
    )


@dataclass(frozen=True, slots=True)
class Histogram:
    """Fixed-width bins over a sample, computed without numpy."""

    edges: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def centers(self) -> tuple[float, ...]:
        return tuple(
            (low + high) / 2 for low, high in zip(self.edges, self.edges[1:], strict=False)
        )

    @property
    def total(self) -> int:
        return sum(self.counts)


def histogram(values: Sequence[float], *, bins: int = 40) -> Histogram:
    """Bin ``values`` into ``bins`` equal-width buckets.

    Degenerate inputs return an empty histogram rather than raising: a figure script
    should not crash because one experiment produced no samples.
    """
    samples = [value for value in values if value == value]  # drop NaN
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not samples:
        return Histogram(edges=(), counts=())

    low, high = min(samples), max(samples)
    if high == low:
        # Every sample in one bucket: make the single bin straddle the value.
        pad = abs(low) * 0.05 or 0.5
        low, high = low - pad, high + pad
    width = (high - low) / bins
    edges = tuple(low + width * index for index in range(bins + 1))
    counts = [0] * bins
    for value in samples:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return Histogram(edges=edges, counts=tuple(counts))


def series_from_rows(rows: Sequence[dict], key: str) -> tuple[float, ...]:
    """Pull one numeric column out of sweep rows, skipping anything non-numeric."""
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Panel:
    """One subplot: named series over a shared x axis, with an optional best point."""

    title: str
    ylabel: str
    series: dict[str, Sequence[float]]
    #: Index to annotate in the *first* series, e.g. the best point of a sweep.
    mark: int | None = None
    mark_text: str = "best"
    log_y: bool = False


@dataclass(frozen=True, slots=True)
class HistogramPanel:
    """One distribution, with an optional reference line such as a p95."""

    title: str
    xlabel: str
    values: Sequence[float]
    ylabel: str = "count"
    reference: float | None = None
    reference_text: str = ""


@dataclass(frozen=True, slots=True)
class Point:
    """A labelled point for the trade-off scatter."""

    label: str
    x: float
    y: float


# ------------------------------------------------------------------- drawing


def _pyplot():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by hand
        raise ModuleNotFoundError(
            "plotting needs matplotlib; install it with: uv pip install -e '.[plot]'"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.autolayout": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.size": 9,
        }
    )
    return plt


def _save(plt, fig, path: str | Path, *, title: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def _finish_axes(ax, panel: Panel) -> None:
    if panel.log_y:
        ax.set_yscale("log")


def plot_timeline(grid: TimelineGrid, path: str | Path, *, title: str = "") -> Path:
    """The scheduler timeline: one row per request, one column per step."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt = _pyplot()
    if not grid.num_requests or not grid.num_steps:
        raise ValueError("nothing to plot: the run has no steps")

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, grid.num_steps * 0.30 + 2.6),
            max(2.4, grid.num_requests * 0.40 + 1.6),
        )
    )
    codes = [[KIND_CODE[kind] for kind in row] for row in grid.kinds]
    ax.imshow(
        codes,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(list(KIND_COLORS)),
        vmin=0,
        vmax=2,
    )

    if grid.num_steps <= 48:
        ax.set_xticks(range(grid.num_steps), [str(i) for i in grid.step_indices])
    else:
        every = max(1, grid.num_steps // 24)
        ticks = list(range(0, grid.num_steps, every))
        ax.set_xticks(ticks, [str(grid.step_indices[i]) for i in ticks])
    ax.set_yticks(range(grid.num_requests), list(grid.request_ids))
    ax.set_xlabel("engine step")
    ax.grid(False)

    if grid.num_steps <= 48 and grid.num_requests <= 24:
        for row in range(grid.num_requests):
            for column in range(grid.num_steps):
                if grid.kinds[row][column]:
                    letter = "P" if grid.kinds[row][column] == "prefill" else "D"
                    ax.text(
                        column,
                        row,
                        f"{letter}{grid.tokens[row][column]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if grid.kinds[row][column] == "decode" else "#1b2a41",
                    )

    ax.legend(
        handles=[
            Patch(facecolor=KIND_COLORS[1], label="prefill chunk (P12 = 12 prefill tokens)"),
            Patch(facecolor=KIND_COLORS[2], label="decode token (D1 = one output token)"),
            Patch(facecolor=KIND_COLORS[0], label="not scheduled"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
        ncols=3,
        frameon=False,
    )
    return _save(plt, fig, path, title=title)


def plot_step_series(series: StepSeries, path: str | Path, *, title: str = "") -> Path:
    """KV pressure and the work mix over the run: three stacked panels."""
    plt = _pyplot()
    if not series.num_steps:
        raise ValueError("nothing to plot: the run has no steps")

    fig, axes = plt.subplots(
        3, 1, figsize=(7.6, 7.4), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    x = series.times_ms

    axes[0].plot(x, [value * 100 for value in series.utilization], color=PALETTE[0], lw=1.6)
    axes[0].fill_between(
        x, [value * 100 for value in series.utilization], color=PALETTE[0], alpha=0.18
    )
    axes[0].plot(
        x,
        [value * 100 for value in series.fragmentation],
        color=PALETTE[3],
        lw=1.2,
        ls="--",
    )
    axes[0].set_ylabel("KV pool (%)")
    axes[0].set_title("how full the KV pool is, and how much of it is tail waste")
    axes[0].legend(["blocks allocated", "reserved but unused"], loc="upper left", ncols=2)

    axes[1].plot(x, series.running, color=PALETTE[2], lw=1.6)
    axes[1].plot(x, series.waiting, color=PALETTE[3], lw=1.6)
    axes[1].set_ylabel("requests")
    axes[1].set_title("who is in the engine: running versus waiting")
    axes[1].legend(["running", "waiting"], loc="upper left", ncols=2)

    axes[2].step(x, series.prefill_tokens, color=PALETTE[0], lw=1.4, where="post")
    axes[2].step(x, series.decode_requests, color=PALETTE[1], lw=1.4, where="post")
    axes[2].set_ylabel("tokens / requests")
    axes[2].set_xlabel("modelled time (ms)")
    axes[2].set_title("one step's work, split between prefill tokens and decode requests")
    axes[2].legend(["prefill tokens", "decode requests"], loc="upper left", ncols=2)

    return _save(plt, fig, path, title=title)


def plot_histograms(
    panels: Sequence[HistogramPanel],
    path: str | Path,
    *,
    title: str = "",
    bins: int = 40,
) -> Path:
    """Latency distributions side by side, each with its tail marked."""
    plt = _pyplot()
    if not panels:
        raise ValueError("nothing to plot: no histogram panels")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), squeeze=False)
    for ax, panel in zip(axes[0], panels, strict=True):
        data = list(panel.values)
        if not data:
            ax.set_title(f"{panel.title} {NO_DATA}")
            continue
        ax.hist(data, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white", lw=0.4)
        if panel.reference is not None:
            ax.axvline(panel.reference, color=PALETTE[3], ls="--", lw=1.3)
            ax.annotate(
                panel.reference_text or f"{panel.reference:,.1f}",
                xy=(panel.reference, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0),
                textcoords="offset points",
                color=PALETTE[3],
                fontsize=8,
            )
        ax.set_title(panel.title)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
    return _save(plt, fig, path, title=title)


def plot_lines(
    x: Sequence[float],
    panels: Sequence[Panel],
    path: str | Path,
    *,
    xlabel: str = "",
    title: str = "",
    log_x: bool = False,
) -> Path:
    """One subplot per metric, sharing an x axis, one line per series."""
    plt = _pyplot()
    if not panels:
        raise ValueError("nothing to plot: no panels")

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7.2, 2.5 * len(panels)), sharex=True, squeeze=False
    )
    for ax, panel in zip(axes[:, 0], panels, strict=True):
        for index, (name, values) in enumerate(panel.series.items()):
            ax.plot(
                x,
                values,
                marker="o",
                ms=4,
                lw=1.6,
                ls=LINESTYLES[index % len(LINESTYLES)],
                color=PALETTE[index % len(PALETTE)],
                label=name,
            )
        if panel.mark is not None and panel.series:
            first = next(iter(panel.series.values()))
            if 0 <= panel.mark < len(first):
                # Headroom, so the label sits inside the axes and not over the title.
                ax.margins(y=0.18)
                ax.annotate(
                    panel.mark_text,
                    xy=(x[panel.mark], first[panel.mark]),
                    xytext=(0, 16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=PALETTE[3],
                    arrowprops={"arrowstyle": "->", "color": PALETTE[3], "lw": 1.0},
                )
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title)
        if len(panel.series) > 1:
            ax.legend(loc="best")
        if log_x:
            ax.set_xscale("log")
        _finish_axes(ax, panel)
    axes[-1, 0].set_xlabel(xlabel)
    return _save(plt, fig, path, title=title)


def plot_bars(
    groups: Sequence[str],
    panels: Sequence[Panel],
    path: str | Path,
    *,
    title: str = "",
    log_x: bool = False,
) -> Path:
    """One subplot per metric, with bars grouped by label."""
    plt = _pyplot()
    if not panels:
        raise ValueError("nothing to plot: no panels")

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.1 * len(panels) + 0.8 * len(groups), 3.8),
        squeeze=False,
    )
    positions = range(len(groups))
    for ax, panel in zip(axes[0], panels, strict=True):
        names = list(panel.series)
        width = 0.8 / max(1, len(names))
        for index, name in enumerate(names):
            offset = (index - (len(names) - 1) / 2) * width
            ax.bar(
                [position + offset for position in positions],
                panel.series[name],
                width=width * 0.92,
                color=PALETTE[index % len(PALETTE)],
                label=name,
            )
        ax.set_xticks(list(positions), list(groups), rotation=18, ha="right")
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title)
        if len(names) > 1:
            ax.legend(loc="best")
        if log_x:
            ax.set_xscale("log")
        _finish_axes(ax, panel)
    return _save(plt, fig, path, title=title)


def plot_scatter(
    points: Sequence[Point],
    path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> Path:
    """A trade-off plane: one labelled point per configuration."""
    plt = _pyplot()
    if not points:
        raise ValueError("nothing to plot: no points")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for index, point in enumerate(points):
        ax.scatter(point.x, point.y, s=70, color=PALETTE[index % len(PALETTE)], zorder=3)
        ax.annotate(
            point.label,
            xy=(point.x, point.y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save(plt, fig, path, title=title)
