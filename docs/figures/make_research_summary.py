"""Build the publication-style summary figure from committed benchmark JSON.

Run with ``python docs/figures/make_research_summary.py`` from the repository
root. The figure is descriptive: simulation measurements do not become
physical-data claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moneyrepair.style import REPORT_PALETTE, load_matplotlib

plt = load_matplotlib()
plt.rcParams["svg.hashsalt"] = "moneyrepair-research-evidence-summary"

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent / "benchmarks"

ALGORITHMS = (
    ("baseline", "Fixed overlap", REPORT_PALETTE[0], "o"),
    ("effectiveness", "Adaptive Etear", REPORT_PALETTE[2], "s"),
    ("effectiveness_gap", "Etear + gap", REPORT_PALETTE[1], "^"),
    ("v43_routed", "Routed v4.3", "#222222", "D"),
)

TRADE_OFF_LABEL_OFFSETS = {
    "baseline": (4, 4),
    "effectiveness": (4, 4),
    "effectiveness_gap": (-72, 8),
    "v43_routed": (8, -1),
}


def _load(name: str) -> dict:
    return json.loads((BENCHMARKS / name).read_text(encoding="utf-8"))


def _series(rows: list[dict], algorithm: str, metric: str) -> tuple[np.ndarray, ...]:
    pieces = np.asarray(sorted({int(row["pieces_per_note"]) for row in rows}))
    mean = []
    deviation = []
    for count in pieces:
        values = [
            float(row[metric])
            for row in rows
            if row["algorithm"] == algorithm and int(row["pieces_per_note"]) == int(count)
        ]
        mean.append(float(np.mean(values)))
        deviation.append(float(np.std(values)))
    return pieces, np.asarray(mean), np.asarray(deviation)


def _mean_at(rows: list[dict], algorithm: str, pieces: int, metric: str) -> float:
    values = [
        float(row[metric]) for row in rows if row["algorithm"] == algorithm and int(row["pieces_per_note"]) == pieces
    ]
    return float(np.mean(values))


def main() -> None:
    v43 = _load("v4_3_geometry_ablation.json")
    v441 = _load("v4_4_1_base_selection_n100_seed7.json")
    rows = v43["rows"]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), constrained_layout=True)

    ax = axes[0, 0]
    for algorithm, label, color, marker in ALGORITHMS:
        x, mean, deviation = _series(rows, algorithm, "automatic_exact_yield")
        ax.errorbar(
            x,
            mean,
            yerr=deviation,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=4.5,
            capsize=2.5,
        )
    ax.set_title("A  Fineness response")
    ax.set_xlabel("pieces per note")
    ax.set_ylabel("automatic exact yield")
    ax.set_xticks([8, 16, 24])
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    ax = axes[0, 1]
    for algorithm, label, color, marker in ALGORITHMS:
        yield_value = _mean_at(rows, algorithm, 24, "automatic_exact_yield")
        precision = _mean_at(rows, algorithm, 24, "automatic_exact_precision")
        false_rate = _mean_at(rows, algorithm, 24, "false_edge_rate")
        size_scale = 1.45 if algorithm == "effectiveness_gap" else 0.72 if algorithm == "v43_routed" else 1.0
        ax.scatter(
            yield_value,
            precision,
            s=(36 + false_rate * 420) * size_scale,
            color=color,
            marker=marker,
            edgecolor="#ffffff",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            label,
            (yield_value, precision),
            xytext=TRADE_OFF_LABEL_OFFSETS[algorithm],
            textcoords="offset points",
            arrowprops=(
                {"arrowstyle": "-", "color": color, "linewidth": 0.7}
                if algorithm == "effectiveness_gap"
                else None
            ),
        )
    ax.set_title("B  Fine-fragment trade-off (p=24)")
    ax.set_xlabel("automatic exact yield")
    ax.set_ylabel("automatic exact precision")
    ax.set_xlim(0.48, 0.96)
    ax.set_ylim(0.79, 1.01)
    ax.grid(color="#E2E2E2", linewidth=0.6)
    ax.text(
        0.02,
        0.04,
        "marker area increases with false-edge rate",
        transform=ax.transAxes,
        color="#59636b",
    )

    ax = axes[1, 0]
    cases = [
        ("Global control", v441["global_control"]),
        ("Disjoint rounds", v441["disjoint_round_robin_intervention"]),
    ]
    metrics = (
        ("oracle_candidate_recall", "oracle recall", REPORT_PALETTE[0]),
        ("exact_yield", "exact yield", REPORT_PALETTE[1]),
        ("exact_precision", "precision", REPORT_PALETTE[2]),
    )
    x = np.arange(len(cases))
    width = 0.23
    for offset, (key, label, color) in enumerate(metrics):
        values = [float(payload[key]) for _, payload in cases]
        bars = ax.bar(x + (offset - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_title("C  Fixed-budget base selection (N=100, p=24)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in cases])
    ax.set_ylabel("fraction")
    ax.set_ylim(0.75, 1.04)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(loc="lower right", ncols=1)

    ax = axes[1, 1]
    stages = [
        ("Deterministic simulation core", "measured", REPORT_PALETTE[2]),
        ("Acquisition coordinate contract", "implemented", REPORT_PALETTE[0]),
        ("Independent physical masks", "pending", "#B8BEC3"),
        ("Physical pose handoff", "pending", "#B8BEC3"),
        ("Real seam / assembly evidence", "pending", "#B8BEC3"),
    ]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, np.ones(len(stages)), color=[color for _, _, color in stages], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([stage for stage, _, _ in stages])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("D  Evidence boundary")
    for row, (_, state, _) in zip(y, stages):
        ax.text(0.98, row, state, ha="right", va="center", color="#10222e", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "MoneyRepair evidence summary\ncommitted simulation benchmarks; physical validation remains pending",
        fontweight="bold",
        y=1.09,
    )
    for suffix in ("png", "svg"):
        metadata = {"Date": None} if suffix == "svg" else {"Software": "MoneyRepair"}
        output_path = HERE / f"research_evidence_summary.{suffix}"
        fig.savefig(
            output_path,
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
        if suffix == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"wrote research evidence summary to {HERE}")


if __name__ == "__main__":
    main()
