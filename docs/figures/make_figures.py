"""Generate the STATUS.md figures from the measured simulation-harness numbers.

Reproducible: ``python docs/figures/make_figures.py``. Every value below is copied
from STATUS.md's measured tables; this script only plots them. No statement here
may exceed STATUS.md. Requires matplotlib (``pip install -e ".[reports]"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from moneyrepair.style import REPORT_PALETTE, load_matplotlib

    plt = load_matplotlib()
except Exception:  # pragma: no cover - standalone fallback
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORT_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

HERE = Path(__file__).resolve().parent
GEO, SERIAL = REPORT_PALETTE[0], REPORT_PALETTE[1]


def _save(fig, name: str) -> None:
    fig.savefig(HERE / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def figure_wall() -> None:
    """The wall: serials rescue precision but not yield; fineness/scale break it."""

    stressors = ["N=100\n(coarse)", "N=200\n(coarse)", "pieces=16\n(fine)", "pieces=24\n(finer)"]
    yield_geo = [0.54, 0.055, 0.10, 0.02]
    yield_ser = [np.nan, 0.26, 0.12, 0.00]
    prec_geo = [0.96, 0.55, 0.45, 0.20]
    prec_ser = [np.nan, 0.98, 1.00, 0.00]
    x = np.arange(len(stressors))
    w = 0.38

    fig, (ax_y, ax_p) = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    for ax, geo, ser, title in (
        (ax_y, yield_geo, yield_ser, "Exact yield"),
        (ax_p, prec_geo, prec_ser, "Precision"),
    ):
        ax.bar(x - w / 2, geo, w, color=GEO, label="geometry only")
        ax.bar(x + w / 2, np.nan_to_num(ser), w, color=SERIAL, label="+ ideal serial anchors")
        for xi, v in zip(x, ser):
            if np.isnan(v):
                ax.text(xi + w / 2, 0.02, "n/a", ha="center", va="bottom", fontsize=6, color="#888")
        ax.set_xticks(x)
        ax.set_xticklabels(stressors)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("fraction")
    ax_y.legend(loc="upper right")
    fig.suptitle("Where the wall is (measured, simulation harness)", fontweight="bold")
    _save(fig, "status_wall")


def figure_scale_vs_fineness() -> None:
    """Scale collapse was a fixable bug; fineness is the real signal wall."""

    fig, (ax_clean, ax_scale) = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)

    notes = ["N=20", "N=50", "N=100"]
    ax_clean.bar(np.arange(len(notes)) - 0.2, [1.0, 1.0, 1.0], 0.4, color=GEO, label="exact yield")
    ax_clean.bar(np.arange(len(notes)) + 0.2, [1.0, 1.0, 1.0], 0.4, color=REPORT_PALETTE[2], label="precision")
    ax_clean.set_xticks(np.arange(len(notes)))
    ax_clean.set_xticklabels(notes)
    ax_clean.set_ylim(0, 1.1)
    ax_clean.set_ylabel("fraction")
    ax_clean.set_title("Clean regime, geometry only\n(coarse pieces)")
    ax_clean.legend(loc="lower center")

    cases = ["N=200\ntight budget\n(pre-fix: crash)", "N=200\ngenerous budget\n(post-fix v4.2.1)", "pieces=16\nany budget"]
    yields = [0.09, 1.00, 0.04]
    colors = [REPORT_PALETTE[3], REPORT_PALETTE[2], REPORT_PALETTE[3]]
    bars = ax_scale.bar(np.arange(len(cases)), yields, 0.6, color=colors)
    ax_scale.set_xticks(np.arange(len(cases)))
    ax_scale.set_xticklabels(cases, fontsize=6.5)
    ax_scale.set_ylim(0, 1.1)
    ax_scale.set_ylabel("exact yield")
    ax_scale.set_title("Scale = compute/bug-bound\nFineness = signal wall")
    for rect, v in zip(bars, yields):
        ax_scale.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Two collapse axes are not the same wall (measured)", fontweight="bold")
    _save(fig, "status_scale_vs_fineness")


def main() -> None:
    figure_wall()
    figure_scale_vs_fineness()
    print(f"wrote figures to {HERE}")


if __name__ == "__main__":
    main()
