from __future__ import annotations

# One palette shared by every Python report panel so multi-panel figures stay
# visually consistent. Colour-blind-safe ordering (blue / orange / green first).
REPORT_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#E45756",
    "#EECA3B",
    "#9D755D",
]

PUBLICATION_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
}


def load_matplotlib():
    """Import matplotlib with the shared publication style applied."""

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("report generation requires matplotlib; install with moneyrepair[reports]") from exc

    mpl.rcParams.update(PUBLICATION_RCPARAMS)
    return plt
