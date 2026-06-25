from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from moneyrepair.style import REPORT_PALETTE, load_matplotlib


@dataclass(frozen=True)
class FigurePanel:
    """One subordinate benchmark/QA panel in a scientific report figure."""

    key: str
    title: str
    kind: str  # "bar" | "grouped_bar" | "line"
    categories: list[str]
    series: dict[str, list[float]]
    ylabel: str = ""
    claim: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"bar", "grouped_bar", "line"}:
            raise ValueError(f"unsupported panel kind: {self.kind}")
        if not self.series:
            raise ValueError(f"panel {self.key} needs at least one data series")
        for name, values in self.series.items():
            if len(values) != len(self.categories):
                raise ValueError(f"panel {self.key} series {name} length must match categories")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_panel(ax, panel: FigurePanel) -> None:
    names = list(panel.series)
    categories = panel.categories
    if panel.kind == "line":
        for offset, name in enumerate(names):
            ax.plot(categories, panel.series[name], marker="o", linewidth=1.4, color=REPORT_PALETTE[offset % len(REPORT_PALETTE)], label=name)
    elif panel.kind == "bar":
        ax.bar(categories, panel.series[names[0]], color=REPORT_PALETTE[: len(categories)])
    else:  # grouped_bar
        positions = np.arange(len(categories))
        width = 0.8 / max(len(names), 1)
        for offset, name in enumerate(names):
            ax.bar(
                positions - 0.4 + width * (offset + 0.5),
                panel.series[name],
                width,
                color=REPORT_PALETTE[offset % len(REPORT_PALETTE)],
                label=name,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(categories)
    ax.set_title(panel.title)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    ax.tick_params(axis="x", rotation=20)
    if len(names) > 1:
        ax.legend()


def _write_source_csv(path: Path, panels: list[FigurePanel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "panel_title", "series", "category", "value"])
        for panel in panels:
            for name, values in panel.series.items():
                for category, value in zip(panel.categories, values):
                    writer.writerow([panel.key, panel.title, name, category, value])


def render_report_figure(
    panels: list[FigurePanel],
    output_prefix: str | Path,
    *,
    title: str = "MoneyRepair report",
    claim: str = "",
    sources: dict[str, str] | None = None,
    dpi: int = 600,
) -> dict:
    """Render a multi-panel figure plus a source CSV and a provenance manifest."""

    if not panels:
        raise ValueError("at least one panel is required")
    plt = load_matplotlib()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    columns = 1 if len(panels) == 1 else 2
    rows = (len(panels) + columns - 1) // columns
    fig = plt.figure(figsize=(3.4 * columns, 2.6 * rows + 0.4), constrained_layout=True)
    grid = fig.add_gridspec(rows, columns)
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[index // columns, index % columns])
        _draw_panel(ax, panel)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    if claim:
        fig.text(0.5, -0.01, claim, ha="center", va="top", fontsize=6, style="italic")

    exports = {
        "svg": str(output_prefix.with_suffix(".svg")),
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "tiff": str(output_prefix.with_suffix(".tiff")),
    }
    fig.savefig(exports["svg"], bbox_inches="tight")
    fig.savefig(exports["pdf"], bbox_inches="tight")
    fig.savefig(exports["tiff"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    data_csv = output_prefix.parent / f"{output_prefix.name}_data.csv"
    _write_source_csv(data_csv, panels)
    exports["data_csv"] = str(data_csv)

    provenance: dict[str, dict] = {}
    for name, source_path in (sources or {}).items():
        record: dict[str, str] = {"path": str(source_path)}
        if Path(source_path).exists():
            record["sha256"] = _sha256(source_path)
        provenance[name] = record

    manifest = {
        "title": title,
        "claim": claim,
        "palette": REPORT_PALETTE,
        "panels": [
            {
                "key": panel.key,
                "title": panel.title,
                "kind": panel.kind,
                "claim": panel.claim,
                "categories": panel.categories,
                "series": list(panel.series),
            }
            for panel in panels
        ],
        "exports": exports,
        "provenance": provenance,
    }
    manifest_path = output_prefix.parent / f"{output_prefix.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["exports"]["manifest"] = str(manifest_path)
    return manifest


def validate_report(manifest: dict) -> list[str]:
    """Report-level QA: flag missing panels, empty series, or missing exports."""

    problems: list[str] = []
    panels = manifest.get("panels", [])
    if not panels:
        problems.append("no panels in figure")
    for panel in panels:
        if not panel.get("title"):
            problems.append(f"panel {panel.get('key', '?')} missing title")
        if not panel.get("series"):
            problems.append(f"panel {panel.get('key', '?')} has no data series")
    exports = manifest.get("exports", {})
    for fmt in ("svg", "pdf", "tiff", "data_csv"):
        path = exports.get(fmt)
        if not path or not Path(path).exists():
            problems.append(f"missing export: {fmt}")
    if not manifest.get("palette"):
        problems.append("missing palette")
    return problems


# --- standard panel builders from toolkit artifacts -------------------------


def algorithm_panel(strategy_results: list[dict]) -> FigurePanel:
    strategies = [str(item["order_strategy"]) for item in strategy_results]
    return FigurePanel(
        key="algorithm",
        title="DFS strategy timing",
        kind="grouped_bar",
        categories=strategies,
        series={
            "solve_s": [float(item["timings_seconds"]["solve"]) for item in strategy_results],
            "matrix_s": [float(item["timings_seconds"]["build_matrix"]) for item in strategy_results],
        },
        ylabel="seconds",
        claim="ordering strategy trades search time against matrix pruning",
    )


def coverage_panel(strategy_results: list[dict]) -> FigurePanel:
    strategies = [str(item["order_strategy"]) for item in strategy_results]
    return FigurePanel(
        key="coverage",
        title="Best coverage",
        kind="line",
        categories=strategies,
        series={"coverage": [float(item.get("best_coverage") or 0.0) for item in strategy_results]},
        ylabel="fraction",
        claim="all strategies reach comparable high coverage",
    )


def footprint_panel(strategy_results: list[dict]) -> FigurePanel:
    footprint = strategy_results[0].get("matrix_footprint", {})
    return FigurePanel(
        key="footprint",
        title="Matrix memory",
        kind="bar",
        categories=["dense_bool", "packed"],
        series={"MB": [float(footprint.get("dense_bool_mb", 0.0)), float(footprint.get("packed_mb", 0.0))]},
        ylabel="MB",
        claim="packed storage is 8x smaller than the dense matrix",
    )


def quality_panel(quality_summaries: dict[str, dict]) -> FigurePanel:
    labels = list(quality_summaries)
    return FigurePanel(
        key="quality",
        title="Acquisition QA",
        kind="grouped_bar",
        categories=labels,
        series={
            "accepted": [int(quality_summaries[label]["accepted"]) for label in labels],
            "rejected": [int(quality_summaries[label]["rejected"]) for label in labels],
        },
        ylabel="frames",
        claim="the contract rejects degraded captures before reconstruction",
    )


def assemble_standard_panels(
    strategy_results: list[dict] | None = None,
    quality_summaries: dict[str, dict] | None = None,
) -> list[FigurePanel]:
    """Build the standard evidence-chain panels from available artifacts."""

    panels: list[FigurePanel] = []
    if quality_summaries:
        panels.append(quality_panel(quality_summaries))
    if strategy_results:
        panels.append(algorithm_panel(strategy_results))
        panels.append(footprint_panel(strategy_results))
        panels.append(coverage_panel(strategy_results))
    if not panels:
        raise ValueError("no artifacts provided to assemble a report figure")
    return panels
