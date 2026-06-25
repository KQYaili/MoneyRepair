from __future__ import annotations

import json
from pathlib import Path

from moneyrepair.style import REPORT_PALETTE, load_matplotlib


def _load_matplotlib():
    return load_matplotlib()


def load_strategy_results(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("strategy benchmark report expects a JSON list")
    return raw


def write_strategy_report(
    results: list[dict],
    output_prefix: str | Path,
    title: str = "MoneyRepair strategy benchmark",
    dpi: int = 600,
) -> dict[str, str]:
    """Create a compact publication-style report from strategy benchmark JSON."""

    if not results:
        raise ValueError("at least one strategy result is required")
    plt = _load_matplotlib()
    strategies = [str(item["order_strategy"]) for item in results]
    solve = [float(item["timings_seconds"]["solve"]) for item in results]
    matrix = [float(item["timings_seconds"]["build_matrix"]) for item in results]
    coverage = [float(item["best_coverage"] or 0.0) for item in results]

    fig = plt.figure(figsize=(6.8, 3.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])

    palette = REPORT_PALETTE
    ax0.bar(strategies, solve, color=palette[: len(strategies)])
    ax0.set_title("DFS solve time")
    ax0.set_ylabel("seconds")
    ax0.tick_params(axis="x", rotation=25)

    ax1.bar(strategies, matrix, color="#9ECAE9")
    ax1.set_title("Matrix build time")
    ax1.set_ylabel("seconds")
    ax1.tick_params(axis="x", rotation=25)

    ax2.plot(strategies, coverage, marker="o", color="#D62728", linewidth=1.4)
    ax2.set_ylim(max(0.0, min(coverage) - 0.02), min(1.0, max(coverage) + 0.02))
    ax2.set_title("Best coverage")
    ax2.set_ylabel("fraction")
    ax2.tick_params(axis="x", rotation=25)

    fig.suptitle(title, fontsize=9, fontweight="bold")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "svg": str(output_prefix.with_suffix(".svg")),
        "pdf": str(output_prefix.with_suffix(".pdf")),
        "tiff": str(output_prefix.with_suffix(".tiff")),
    }
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs
