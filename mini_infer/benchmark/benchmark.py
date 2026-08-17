"""Running benchmarks and reporting their results.

A benchmark is a workload plus an engine configuration. Sweeping one axis (arrival
rate, KV pool size, block size, policy) and comparing the resulting rows is what
turns the runtime into an engineering artifact rather than a demo.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from mini_infer.benchmark.driver import Driver, RunResult
from mini_infer.benchmark.workloads import WorkloadSpec
from mini_infer.config import EngineConfig
from mini_infer.engine.engine import Engine


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One labelled point in a sweep."""

    label: str
    workload: WorkloadSpec
    config: EngineConfig

    def run(self, *, max_steps: int | None = None) -> RunResult:
        engine = Engine(self.config)
        return Driver(engine, self.workload.build(), max_steps=max_steps).run()


@dataclass(frozen=True, slots=True)
class SweepRow:
    """A run together with the label and configuration that produced it."""

    label: str
    config: EngineConfig
    result: RunResult

    @property
    def metrics(self):
        return self.result.metrics

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {"label": self.label}
        row.update(self.config.to_dict())
        row.pop("runner", None)
        row.update(self.result.as_row())
        return row


def run_case(
    workload: WorkloadSpec,
    config: EngineConfig,
    *,
    max_steps: int | None = None,
    label: str = "",
) -> RunResult:
    return BenchmarkCase(label=label or config.policy, workload=workload, config=config).run(
        max_steps=max_steps
    )


def sweep(
    workload: WorkloadSpec,
    configs: Iterable[EngineConfig],
    *,
    label: Callable[[EngineConfig], str] | None = None,
    max_steps: int | None = None,
) -> list[SweepRow]:
    """Run the same workload against several engine configurations."""
    rows = []
    for config in configs:
        case_label = label(config) if label is not None else config.policy
        result = BenchmarkCase(
            label=case_label, workload=workload, config=config
        ).run(max_steps=max_steps)
        rows.append(SweepRow(label=case_label, config=config, result=result))
    return rows


def sweep_axis(
    workload: WorkloadSpec,
    base: EngineConfig,
    *,
    values: Sequence[Any],
    apply: Callable[[EngineConfig, Any], EngineConfig],
    label: Callable[[Any], str] = str,
    max_steps: int | None = None,
) -> list[SweepRow]:
    """Sweep one configuration axis, e.g. ``num_blocks`` over a list of pool sizes."""
    configs = [apply(base, value) for value in values]
    labels = {id(config): label(value) for config, value in zip(configs, values, strict=True)}
    return sweep(workload, configs, label=lambda cfg: labels[id(cfg)], max_steps=max_steps)


# ----------------------------------------------------------------- reporting


def results_to_rows(rows: Sequence[SweepRow]) -> list[dict[str, object]]:
    """One flat record per run, labelled by workload and configuration."""
    return [row.as_row() for row in rows]


@dataclass(frozen=True, slots=True)
class Report:
    """Columns worth printing, in a readable order."""

    columns: tuple[str, ...] = (
        "label",
        "policy",
        "arrival",
        "completed",
        "output_tok_per_s",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "mean_tpot_ms",
        "p95_itl_ms",
        "p95_e2e_ms",
        "mean_queue_ms",
        "peak_kv_utilization",
        "mean_fragmentation",
        "max_decode_batch",
        "preemptions",
    )

    #: Readable header for each column. A ClassVar so it is not a dataclass field.
    HEADERS: ClassVar[dict[str, str]] = {
        "label": "case",
        "policy": "policy",
        "arrival": "arrival",
        "completed": "done",
        "output_tok_per_s": "out tok/s",
        "mean_ttft_ms": "TTFT mean",
        "p95_ttft_ms": "TTFT p95",
        "mean_tpot_ms": "TPOT mean",
        "p95_itl_ms": "ITL p95",
        "p95_e2e_ms": "E2E p95",
        "mean_queue_ms": "queue mean",
        "peak_kv_utilization": "KV peak",
        "mean_fragmentation": "frag mean",
        "max_decode_batch": "decode batch",
        "preemptions": "evictions",
    }

    def render(self, rows: Sequence[dict[str, object]], title: str = "") -> str:
        """Render rows as a fixed-width table, omitting empty columns."""
        present = [c for c in self.columns if any(c in row for row in rows)]
        if not present:
            return "(no columns)"
        headers = [self.HEADERS.get(c, c) for c in present]
        cells = [[_format(row.get(c)) for c in present] for row in rows]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(present))
        ]
        lines = []
        if title:
            lines.append(title)
        lines.append("  ".join(h.rjust(widths[i]) for i, h in enumerate(headers)))
        lines.append("  ".join("-" * w for w in widths))
        for row in cells:
            lines.append("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))
        return "\n".join(lines)


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# -------------------------------------------------------------------- export


def write_json(rows: Sequence[SweepRow], path: str | Path) -> Path:
    """Write every run with its workload and configuration, so it can be replayed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {
                "label": row.label,
                "config": row.config.to_dict(),
                "workload": row.result.workload.describe(),
                "metrics": row.metrics.as_row(),
                "stalled": row.result.stalled,
                "steps": row.result.steps,
            }
            for row in rows
        ]
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def write_csv(rows: Sequence[SweepRow], path: str | Path) -> Path:
    """Write every run as one CSV row, unioning the keys across runs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = results_to_rows(rows)
    if not records:
        target.write_text("")
        return target
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
