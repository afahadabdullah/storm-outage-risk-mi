"""Shared, print-safe styling for Phase 2 result figures."""
from __future__ import annotations

from pathlib import Path

INK = "#17232D"
MUTED = "#5D6B76"
FAINT = "#9AA6AE"
GRID = "#DDE4E8"
ACCENT = "#12626F"
REFERENCE = "#91A0AA"
OBSERVED = "#A43A25"
SEQUENCE = ["#D7E9EB", "#A8D0D5", "#6DABB4", "#347D89", "#124F5B"]


def apply() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter})  {title}", loc="left")


def save(fig, png_path: Path, note: str = "") -> tuple[Path, Path]:
    """Save a 300-dpi PNG and matching vector PDF."""
    import matplotlib.pyplot as plt

    if note:
        fig.text(0.005, 0.003, note, color=FAINT, fontsize=7, ha="left", va="bottom")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path
