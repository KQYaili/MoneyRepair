"""Render the v4.3.1 mechanism-decomposition figure from benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moneyrepair.style import load_matplotlib

plt = load_matplotlib()

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "benchmarks" / "v4_3_geometry_ablation.json"

BLUE = "#4C78A8"
GREEN = "#54A24B"
ORANGE = "#F58518"
RED = "#E45756"
BLACK = "#222222"
GRAY = "#B9B9B9"


def _summary_index(payload: dict) -> dict[tuple[int, str], dict]:
    return {
        (int(row["pieces_per_note"]), str(row["algorithm"])): row
        for row in payload["summary"]
    }


def _annotate_bars(ax, bars, *, digits: int = 1) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def main() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    summary = _summary_index(payload)
    config = payload["config"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)

    # A: what the adaptive edge gate removes at the hard p=24 regime.
    ax = axes[0, 0]
    algorithms = ("baseline", "effectiveness")
    labels = ("Fixed overlap", "Adaptive Etear")
    true_edges = [summary[(24, name)]["mean_true_accepted_edges"] for name in algorithms]
    false_edges = [summary[(24, name)]["mean_false_accepted_edges"] for name in algorithms]
    x = np.arange(len(algorithms))
    true_bars = ax.bar(x, true_edges, color=BLUE, label="true accepted edges")
    false_bars = ax.bar(
        x,
        false_edges,
        bottom=true_edges,
        color=RED,
        label="false accepted edges",
    )
    _annotate_bars(ax, true_bars)
    for bar, base, value in zip(false_bars, true_edges, false_edges):
        ax.annotate(
            f"false {value:.1f}",
            (bar.get_x() + bar.get_width() / 2, base + value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color=RED,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("accepted edge count")
    ax.set_title("A  Etear filters the edge graph (p=24)")
    ax.legend(loc="upper right")

    # B: the group stage changes both recall and final selection precision.
    ax = axes[0, 1]
    algorithms = ("effectiveness", "effectiveness_gap")
    labels = ("Etear", "Etear + group gap")
    yield_values = [
        summary[(24, name)]["mean_automatic_exact_yield"] for name in algorithms
    ]
    precision_values = [
        summary[(24, name)]["mean_automatic_exact_precision"]
        for name in algorithms
    ]
    x = np.arange(len(algorithms))
    width = 0.34
    yield_bars = ax.bar(
        x - width / 2,
        yield_values,
        width,
        color=GREEN,
        label="automatic exact yield",
    )
    precision_bars = ax.bar(
        x + width / 2,
        precision_values,
        width,
        color=ORANGE,
        label="automatic exact precision",
    )
    _annotate_bars(ax, yield_bars, digits=3)
    _annotate_bars(ax, precision_bars, digits=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("fraction")
    ax.set_title("B  Assembly context recovers and reranks (p=24)")
    ax.legend(loc="lower right")

    # C: generated gap candidates are not equivalent to selected evidence.
    ax = axes[1, 0]
    pieces = np.asarray([8, 16, 24])
    generated = np.asarray(
        [summary[(int(p), "effectiveness_gap")]["mean_gap_candidates"] for p in pieces]
    )
    selected = np.asarray(
        [
            summary[(int(p), "effectiveness_gap")]["mean_selected_gap_candidates"]
            for p in pieces
        ]
    )
    partial = np.asarray(
        [
            summary[(int(p), "effectiveness_gap")][
                "mean_selected_partial_gap_candidates"
            ]
            for p in pieces
        ]
    )
    bars = ax.bar(pieces, generated, width=3.8, color=GRAY, label="generated gap candidates")
    _annotate_bars(ax, bars)
    ax.set_xlabel("pieces per note")
    ax.set_ylabel("generated candidates")
    ax.set_xticks(pieces)
    ax.set_title("C  Gap evidence is selective")
    selected_axis = ax.twinx()
    selected_axis.plot(
        pieces,
        selected,
        color=BLACK,
        marker="o",
        linewidth=1.6,
        label="selected gap candidates",
    )
    selected_axis.plot(
        pieces,
        partial,
        color=ORANGE,
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label="selected from partial cores",
    )
    selected_axis.set_ylabel("selected candidates")
    selected_axis.set_ylim(0.0, max(3.2, float(selected.max()) * 1.2))
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = selected_axis.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")

    # D: routing avoids the heavy path where fixed overlap is already reliable.
    ax = axes[1, 1]
    always_gap = np.asarray(
        [summary[(int(p), "effectiveness_gap")]["mean_elapsed_seconds"] for p in pieces]
    )
    routed = np.asarray(
        [summary[(int(p), "v43_routed")]["mean_elapsed_seconds"] for p in pieces]
    )
    ax.plot(
        pieces,
        always_gap,
        color=ORANGE,
        marker="^",
        linewidth=1.6,
        label="always Etear + gap",
    )
    ax.plot(
        pieces,
        routed,
        color=BLACK,
        marker="D",
        linewidth=1.6,
        linestyle="--",
        label="v4.3 routed",
    )
    ax.set_yscale("log")
    ax.set_xticks(pieces)
    ax.set_xlabel("pieces per note")
    ax.set_ylabel("seconds (log scale)")
    ax.set_title("D  Routing avoids unnecessary heavy search")
    ax.legend(loc="upper left")

    for ax in axes.flat:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(
        "MoneyRepair v4.3.1 mechanism decomposition\n"
        f"placed-fragment simulation, N={config['notes']}, seeds={config['seeds']}",
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            HERE / f"v4_3_1_mechanism_decomposition.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"wrote v4.3.1 mechanism figure to {HERE}")


if __name__ == "__main__":
    main()
