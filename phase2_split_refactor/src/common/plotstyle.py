"""One visual system for every Phase 2 figure.

Figures that go into a write-up together should look like they came from the
same study. This module is the single place that decides that, so a new figure
inherits the decisions instead of re-inventing them.

Colour is assigned by the JOB the colour does, not by taste:

  sequential  magnitude, one direction        single hue, light -> dark
  diverging   polarity around a meaningful 0  two hues + a NEUTRAL midpoint
  categorical identity (regime labels)        fixed order, never cycled
  status      good / warning / bad            reserved, never reused as a series

Two rules are load-bearing rather than decorative:

* **No red-green diverging ramp.** `RdYlGn` is the default reach for a skill map
  and it is the single worst choice for the ~8% of male readers with a red-green
  deficiency: positive and negative skill become the same colour. The diverging
  ramp here is orange-to-blue through a near-neutral midpoint, which survives
  deuteranopia, protanopia and greyscale printing.
* **Zero is pinned to the neutral midpoint** on every diverging map, via
  `diverging_norm`. A diverging ramp whose midpoint drifts off zero is worse than
  a sequential one, because it *looks* like it encodes sign and does not.

The categorical set below was validated (not eyeballed) for lightness band,
chroma floor, adjacent-pair CVD separation, normal-vision separation and
contrast against a light surface. `benign` is deliberately NOT a categorical
slot -- it is the absence of a hazard regime, so it takes neutral grey.

These are print/report figures, so they commit to a light surface rather than
carrying a second dark theme.
"""
from __future__ import annotations

from pathlib import Path

# ---- ink and surface --------------------------------------------------------
INK = "#16202B"          # primary text
MUTED = "#5C6B79"        # secondary text, axis labels
FAINT = "#93A1AC"        # tertiary, annotations
GRID = "#DCE3E8"         # recessive gridlines
SURFACE = "#FFFFFF"
PANEL = "#F4F7F9"        # subtle panel fill where one is needed
MISSING = "#E4E9ED"      # counties with no data on a map

# ---- accents ----------------------------------------------------------------
ACCENT = "#12626F"       # the study's own hue: model results
REFERENCE = "#8A97A3"    # baselines and reference models, deliberately recessive

# ---- status (reserved; never used as a series colour) -----------------------
GOOD = "#2A6647"
WARN = "#9A6410"
BAD = "#A8321C"

# ---- categorical: hazard regimes, fixed order, never cycled -----------------
REGIME_COLORS = {
    "convective_wind": "#C05E17",   # warm: buoyancy-driven
    "synoptic_wind": "#31699F",     # cool: large-scale pressure gradient
    "ice": "#8A4FBE",
    "wet_snow": "#94640F",
    "benign": "#B7C1C9",            # neutral: no hazard regime, not a 5th hue
}

# ---- ramps ------------------------------------------------------------------
# Sequential: single hue, light -> dark, monotone in lightness.
SEQ_HEX = ["#F2F7F8", "#CFE3E6", "#A5CBD1", "#6FADB6", "#3D8994", "#1D6674", "#0B4550"]
# Diverging: orange <- neutral -> blue. CVD-safe; greyscale-safe.
DIV_HEX = ["#8C4308", "#C0711F", "#E0A86B", "#EFEFEC", "#8FB3CB", "#3E7BA8", "#12405F"]


def sequential():
    """Magnitude ramp: single hue, light to dark."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("storm_seq", SEQ_HEX)


def diverging():
    """Polarity ramp: two hues through a neutral midpoint. Never red-green."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("storm_div", DIV_HEX)


def diverging_norm(values, center: float = 0.0, robust: float = 0.98):
    """Symmetric norm that pins `center` to the ramp's neutral midpoint.

    Uses a high quantile rather than the raw extreme so one pathological county
    cannot flatten the whole map to its midpoint. Returns None when the field is
    degenerate, so the caller falls back to a plain linear scale instead of
    drawing a diverging ramp that encodes nothing.
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    finite = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return None
    span = float(np.quantile(np.abs(finite - center), robust))
    if not np.isfinite(span) or span <= 0:
        return None
    return TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)


def apply_style() -> None:
    """rcParams for every figure in the study. Call once, at import of a script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        # Recessive frame: keep the two axes the reader needs, drop the box.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def panel_label(ax, letter: str, title: str = "") -> None:
    """Consistent (a) (b) (c) panel tags for a multi-panel publication figure.

    The letter goes INSIDE the title string rather than as floating text above
    it: a separate text artist at transAxes y>1 collides with the title on any
    figure whose panels are not tall, and the collision only shows up once the
    figure is rendered.
    """
    ax.set_title(f"({letter})  {title}".rstrip(), loc="left")


def map_axes(ax) -> None:
    """A choropleth is a picture, not a plot: no frame, no grid, equal aspect."""
    ax.set_axis_off()
    ax.set_aspect("equal")


def footnote(fig, text: str) -> None:
    """One recessive provenance line. Every published figure should carry one."""
    fig.text(0.005, 0.002, text, ha="left", va="bottom", fontsize=7,
             color=FAINT, wrap=True)


def save(fig, path: Path, note: str = "") -> Path:
    """Write the figure and return the path, with an optional provenance note."""
    if note:
        footnote(fig, note)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
