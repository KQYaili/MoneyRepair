"""Plot the measured v4.3 geometry ablation from its JSON artifact.

Run with ``python docs/figures/make_v43_ablation.py`` after generating
``docs/benchmarks/v4_3_geometry_ablation.json``. Requires the ``reports`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moneyrepair.style import load_matplotlib

plt = load_matplotlib()

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "benchmarks" / "v4_3_geometry_ablation.json"

ALGORITHMS = (
    ("baseline", "Fixed overlap", "#4C78A8", "o", "-"),
    ("effectiveness", "Adaptive Etear", "#54A24B", "s", "-"),
    ("effectiveness_gap", "Etear + group gap", "#F58518", "^", "-"),
    ("v43_routed", "Complexity routed", "#222222", "D", "--"),
)


def _series(
    rows: list[dict], algorithm: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pieces = sorted({int(row["pieces_per_note"]) for row in rows})
    means: list[float] = []
    deviations: list[float] = []
    for piece_count in pieces:
        values = [
            float(row[metric])
            for row in rows
            if row["algorithm"] == algorithm
            and int(row["pieces_per_note"]) == piece_count
        ]
        means.append(float(np.mean(values)))
        deviations.append(float(np.std(values)))
    return np.asarray(pieces), np.asarray(means), np.asarray(deviations)


def _panel(
    ax,
    rows: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    *,
    log_scale: bool = False,
    ylim: tuple[float, float] = (0.0, 1.05),
) -> None:
    for algorithm, label, color, marker, linestyle in ALGORITHMS:
        x, mean, deviation = _series(rows, algorithm, metric)
        ax.errorbar(
            x,
            mean,
            yerr=deviation,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=4.5,
            capsize=2.5,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("pieces per note")
    ax.set_ylabel(ylabel)
    ax.set_xticks([8, 16, 24])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(*ylim)


def main() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    rows = payload["rows"]
    config = payload["config"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), constrained_layout=True)
    _panel(
        axes[0, 0],
        rows,
        "automatic_exact_yield",
        "A  Automatic exact yield",
        "fraction",
    )
    _panel(
        axes[0, 1],
        rows,
        "automatic_exact_precision",
        "B  Automatic exact precision",
        "fraction",
    )
    _panel(
        axes[1, 0],
        rows,
        "false_edge_rate",
        "C  Accepted false-edge rate",
        "fraction",
        ylim=(0.0, 0.12),
    )
    _panel(
        axes[1, 1],
        rows,
        "elapsed_seconds",
        "D  End-to-end trial cost",
        "seconds (log scale)",
        log_scale=True,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncols=4,
        frameon=False,
    )
    fig.suptitle(
        "MoneyRepair v4.3 geometry ablation\n"
        f"placed-fragment simulation, N={config['notes']}, seeds={config['seeds']}",
        fontweight="bold",
        y=1.10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            HERE / f"v4_3_geometry_ablation.{suffix}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"wrote v4.3 ablation figure to {HERE}")


if __name__ == "__main__":
    main()
