# -*- coding: utf-8 -*-
"""Shared chart style for the data figures (Fig. 8-11).

The schematic figures (Fig. 1-7) are hand-written SVG; the data figures are
matplotlib.  This module is the single place where the two families are kept
visually consistent: same palette, same sans face, same hairline weight, same
type floor.

Contract:
  * palette identical to gen_fig1_arch_v3.py -- contribution accents
    BLUE_C1 / GREEN_C2 / ORANGE_C3, neutral greys C_LIGHT / C_GRAY, ink INK.
  * one-column IEEE figure is 3.5 in wide; two-column is 7.16 in.  Type floor
    is 7 pt AT FINAL SIZE, so nothing below 7 pt may be set here.
  * no chartjunk: no top/right spine, no vertical grid unless the axis is
    logarithmic, no legend frame, no bold except the emphasised series.
  * every figure saves .svg (vector, for LaTeX) and .png at 300 dpi.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------- palette ---
INK = '#141414'
C_LIGHT = '#efefef'
C_GRAY = '#b8b8b8'
C_MID = '#8a8a8a'
C_RED = '#a83a32'
C_GOLD = '#e0a020'
BLUE_C1 = '#1f6feb'      # contribution 1
GREEN_C2 = '#2da44e'     # contribution 2
ORANGE_C3 = '#e36209'    # contribution 3
LEAD = '#c0392b'         # the only red used for leader notes
OLIVE = '#6b8e23'

COL1 = 3.5               # IEEE one-column width, inches
COL2 = 7.16              # IEEE two-column width, inches

FS_TITLE = 9.0
FS_LABEL = 8.0
FS_TICK = 7.5
FS_NOTE = 7.0            # type floor -- nothing smaller


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_NOTE,
        "axes.edgecolor": C_MID,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": C_MID,
        "ytick.color": C_MID,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": C_LIGHT,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "svg.fonttype": "none",
    })


def strip(ax, left=True, bottom=True):
    """Remove the top/right spines; keep tick labels dark enough to read."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(labelcolor=INK)


def save(fig, stem):
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    for ext in ("svg", "png"):
        p = os.path.join(d, "%s.%s" % (stem, ext))
        fig.savefig(p)
        print("wrote", p)
