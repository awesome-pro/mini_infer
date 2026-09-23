"""Text rendering of an engine run.

Produces the scheduler timeline (which request got how much work in which step)
and a KV pool occupancy strip, both readable in a terminal.
"""

from __future__ import annotations

from collections.abc import Sequence

from miniserve.engine.engine import Engine, EngineStep
from miniserve.engine.request import Request
from miniserve.engine.scheduler import ScheduledKind

FILLED = "#"
EMPTY = "."


def work_cell(step: EngineStep, request_id: str) -> str:
    """``P<n>`` for a prefill chunk of n tokens, ``D1`` for a decode token."""
    tokens = 0
    kind: ScheduledKind | None = None
    for work in step.output.work:
        if work.request_id == request_id:
            tokens += work.num_new_tokens
            kind = work.kind
    if kind is None or tokens == 0:
        return "-"
    return f"{'P' if kind is ScheduledKind.PREFILL else 'D'}{tokens}"


def timeline_lines(
    steps: Sequence[EngineStep],
    requests: Sequence[Request],
    *,
    max_rows: int | None = None,
    show_time: bool = True,
) -> list[str]:
    """Render the step-by-step timeline as lines of text."""
    ids = [r.id for r in requests]
    width = max((len(i) for i in ids), default=3) + 2
    header = (f"{'time':>8} " if show_time else "") + "".join(f"{i:>{width}}" for i in ids)
    header += "   batch"

    lines = [header, "-" * len(header)]
    shown = steps if max_rows is None else steps[:max_rows]
    for step in steps if max_rows is None else shown:
        row = f"{step.end_time:>8.3f} " if show_time else ""
        row += "".join(f"{work_cell(step, i):>{width}}" for i in ids)
        row += f"   {step.output.num_scheduled_tokens:>4} tok {step.duration * 1000:>8.2f} ms"
        lines.append(row)

    if max_rows is not None and len(steps) > max_rows:
        lines.append(f"{'...':>8} {len(steps) - max_rows} more steps")
    lines.append("-" * len(header))
    totals = "".join(f"{r.num_generated:>{width}}" for r in requests)
    lines.append((f"{'gen':>8} " if show_time else "") + totals)
    return lines


def kv_pool_lines(engine: Engine, *, samples: int = 8) -> list[str]:
    """Render KV occupancy over the run as a strip of filled/empty blocks."""
    memory = engine.memory
    if memory is None:
        return ["(no KV manager attached)"]

    total = memory.num_blocks
    lines = []
    for step in _sampled_steps(engine, samples):
        free = step.free_blocks if step.free_blocks is not None else total
        used = total - free
        bar = FILLED * used + EMPTY * free
        ratio = used / total if total else 0.0
        lines.append(
            f"step {step.index:>4}  [{bar}]  {used:>3}/{total} blocks ({ratio:6.1%})"
        )
    return lines


def _sampled_steps(engine: Engine, samples: int) -> list[EngineStep]:
    history = engine.history
    if not history:
        return []
    if len(history) <= samples:
        return list(history)
    stride = max(1, len(history) // samples)
    picked = list(history[::stride])[:samples]
    if picked[-1] is not history[-1]:
        picked.append(history[-1])
    return picked
