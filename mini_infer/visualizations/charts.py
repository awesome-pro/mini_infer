"""Text rendering of a sweep: the shape of a trade-off without leaving the terminal.

A curve is what makes a scheduling trade-off legible, and text is deliberately the
primary renderer: it works in a terminal, in a README diff, in CI output and in a
code review, none of which a PNG does.

Character grids cannot place arbitrary labels without them colliding, so each metric
gets its own row-per-label bar chart rather than a shared scatter.
"""

from __future__ import annotations

from collections.abc import Sequence

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    """A one-line density plot, for a metric that only needs its shape."""
    finite = [v for v in values if v == v]
    if not finite:
        return ""
    low, high = min(finite), max(finite)
    if high == low:
        return BLOCKS[0] * len(finite)
    span = high - low
    last = len(BLOCKS) - 1
    return "".join(
        BLOCKS[min(last, int((value - low) / span * last))] for value in finite
    )


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    width: int = 36,
    title: str = "",
    unit: str = "",
    bar: str = "\u2588",
) -> str:
    """One horizontal bar per label, scaled to the largest value.

    Values are printed numerically as well as drawn, so the chart can be read at any
    terminal width and copied into a README without losing information.
    """
    if not labels:
        return "(no data)"
    finite = [v for v in values if v == v]
    if not finite:
        return "(no data)"

    lines = [title] if title else []
    label_width = max(len(str(label)) for label in labels)
    texts = [_format(value) for value in values]
    value_width = max(len(text) for text in texts)
    top = max(finite) or 1.0

    for label, value, text in zip(labels, values, texts, strict=True):
        if value != value:  # NaN: the metric is undefined for this run
            lines.append(f"  {str(label):>{label_width}}  {'':<{width}} {'-':>{value_width}}")
            continue
        length = max(1, int(round(value / top * width))) if value > 0 else 0
        lines.append(
            f"  {str(label):>{label_width}}  {bar * length:<{width}} "
            f"{text:>{value_width}}{unit}"
        )
    return "\n".join(lines)


def metric_panel(
    labels: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    width: int = 36,
    units: dict[str, str] | None = None,
    title: str = "",
) -> str:
    """Several metrics over one swept axis, each scaled to its own maximum.

    Scaling per metric is the point: putting an absolute latency and an absolute
    throughput on one axis shows only that they have different units. What matters is
    how each responds to the sweep.
    """
    if not labels:
        return "(no data)"
    blocks = [title] if title else []
    units = units or {}
    for name, values in series.items():
        blocks.append(bars(labels, values, width=width, title=f"{name}", unit=units.get(name, "")))
    return "\n".join(blocks)


def trend_summary(labels: Sequence[str], series: dict[str, Sequence[float]]) -> str:
    """Every series as a sparkline on one line, for comparing directions of travel."""
    if not labels:
        return "(no data)"
    lines = []
    for name, values in series.items():
        line = sparkline(values)
        if not line:
            continue
        finite = [v for v in values if v == v]
        lines.append(
            f"  {name:<18} {line}   {_format(min(finite))} .. {_format(max(finite))}"
        )
    if len(labels) <= 24:
        lines.append(f"  {'axis':<18} {' '.join(str(label)[:1] for label in labels)}")
    return "\n".join(lines)


def _format(value: float) -> str:
    if value != value:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")
