"""Plot the measured v4.3 same-seed A/B benchmark as static docs figures.

Reproducible: ``python docs/figures/make_v43_ab_figures.py``. The means below are
copied from the same-seed A/B run summarised in ``runs/v43_ab_report.json``
(mean over seeds 7, 8, 9; pieces per note 8 / 16 / 24). This script only plots
them; it does not re-run the benchmark. Requires matplotlib
(``pip install -e ".[reports]"``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display required

import numpy as np  # noqa: E402

try:  # keep panel colours consistent with the other report figures
    from moneyrepair.style import PUBLICATION_RCPARAMS, REPORT_PALETTE

    matplotlib.rcParams.update(PUBLICATION_RCPARAMS)
except Exception:  # pragma: no cover - standalone fallback
    REPORT_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2"]

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

# Categories shared by both figures.
PIECES = ["p=8", "p=16", "p=24"]

# Mean over seeds 7, 8, 9 (from runs/v43_ab_report.json).
YIELD = {
    "baseline": [1.000, 1.000, 1.000],
    "v43_routed": [1.000, 1.000, 0.933],
}
FALSE_EDGE_RATE = {
    "baseline": [0.053, 0.045, 0.046],
    "effectiveness_gap": [0.000, 0.002, 0.017],
}

# Algorithm identity colours mirror docs/figures/make_v43_ablation.py.
BASELINE = "#4C78A8"
EFFECTIVENESS_GAP = "#F58518"
V43_ROUTED = "#222222"


def _save(fig, name: str) -> None:
    fig.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _grouped_bars(
    ax,
    left_series: tuple[str, list[float], str],
    right_series: tuple[str, list[float], str],
    *,
    ylabel: str,
    ylim: tuple[float, float],
    value_fmt: str,
    legend_loc: str = "upper right",
) -> None:
    x = np.arange(len(PIECES))
    w = 0.38
    for offset, (label, values, color) in (
        (-w / 2, left_series),
        (w / 2, right_series),
    ):
        bars = ax.bar(x + offset, values, w, color=color, label=label)
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                value + ylim[1] * 0.015,
                format(value, value_fmt),
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(PIECES)
    ax.set_xlabel("pieces per note")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc=legend_loc)


def figure_yield() -> None:
    """Automatic exact yield: baseline vs complexity-routed v4.3."""

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    _grouped_bars(
        ax,
        ("baseline", YIELD["baseline"], BASELINE),
        ("v43_routed", YIELD["v43_routed"], V43_ROUTED),
        ylabel="automatic exact yield (fraction)",
        ylim=(0.0, 1.1),
        value_fmt=".3f",
        legend_loc="lower center",
    )
    ax.set_title("Automatic exact yield")
    fig.suptitle(
        "MoneyRepair v4.3 A/B: baseline vs routed\n"
        "same-seed simulation, N=10, seeds=[7, 8, 9]",
        fontweight="bold",
    )
    _save(fig, "v4_3_ab_yield")


def figure_false_edge_rate() -> None:
    """Accepted false-edge rate: baseline vs Etear + group gap."""

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    _grouped_bars(
        ax,
        ("baseline", FALSE_EDGE_RATE["baseline"], BASELINE),
        (
            "effectiveness_gap",
            FALSE_EDGE_RATE["effectiveness_gap"],
            EFFECTIVENESS_GAP,
        ),
        ylabel="accepted false-edge rate (fraction)",
        ylim=(0.0, 0.07),
        value_fmt=".3f",
    )
    ax.set_title("Accepted false-edge rate")
    fig.suptitle(
        "MoneyRepair v4.3 A/B: baseline vs effectiveness gap\n"
        "same-seed simulation, N=10, seeds=[7, 8, 9]",
        fontweight="bold",
    )
    _save(fig, "v4_3_ab_false_edge_rate")


def main() -> None:
    figure_yield()
    figure_false_edge_rate()
    print(f"wrote v4.3 A/B figures to {HERE}")


if __name__ == "__main__":
    main()
