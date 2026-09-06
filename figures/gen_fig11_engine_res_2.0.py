# -*- coding: utf-8 -*-
"""
Figure 8 of the article: how the nine engine classes divide the two scarce
resources, LUT and DSP, drawn as two Vivado "Utilization" panels.

Look and colours are taken from the Vivado Project Summary utilization graph
(pixel-sampled from the Xilinx university-program lab screenshot,
xilinx.github.io/xup_fpga_vivado_flow/images/lab1/Fig37.png):
  bar fill        #73e66a   (edge #a1ec9c)
  panel ground    #f6f6f6
  header strip    #ededed, bold black title
  axis / labels   #404040
  bar value text  #808080, bold, "NN%"
  grid lines      #bebebe at 0 / 25 / 50 / 75 / 100
Both panels are shares of the nine-engine totals (133,919 LUT, 849 DSP), so
they are directly comparable; the underlying counts are in Table 2.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fig_style import COL1, save, use_style

use_style()

MUL = "×"
ENGINES = [
    ("Fused attn.",          33711,  80),
    ("Conv 3%s3" % MUL,      24318, 576),
    ("Conv 1%s1" % MUL,      17211, 128),
    ("DW conv",              16215,  17),
    ("Add",                  14489,  32),
    ("Concat",               14195,  16),
    ("SPPF",                 12138,   0),
    ("Upsample / Focus", 917 + 725,   0),
]
LUT_TOT, DSP_TOT = 133919, 849
lut_sh = [100.0 * e[1] / LUT_TOT for e in ENGINES]
dsp_sh = [100.0 * e[2] / DSP_TOT for e in ENGINES]
names = [e[0] for e in ENGINES]

# ---- Vivado palette (sampled) ----------------------------------------------
BAR, BAR_EDGE = "#73e66a", "#a1ec9c"
GROUND, HEADER = "#f6f6f6", "#ededed"
INK, VALUE, GRID = "#404040", "#808080", "#bebebe"

FS_TICK, FS_LABEL, FS_VALUE, FS_HEAD = 7.0, 7.5, 7.0, 7.5
n = len(ENGINES)

fig = plt.figure(figsize=(COL1 + 0.6, 2.45))
fig.patch.set_facecolor("white")
# two panels; the left one carries the row labels
L, R, TOP, BOT, GAP = 0.30, 0.995, 0.86, 0.19, 0.06
W = (R - L - GAP) / 2.0
axL = fig.add_axes([L, BOT, W, TOP - BOT])
axD = fig.add_axes([L + W + GAP, BOT, W, TOP - BOT])


def panel(ax, vals, title):
    ax.set_facecolor(GROUND)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, n - 0.4)
    for x in (25, 50, 75, 100):
        ax.axvline(x, color=GRID, lw=0.6, zorder=1)
    ys = list(range(n))[::-1]
    ax.barh(ys, vals, height=0.52, color=BAR, edgecolor=BAR_EDGE, lw=0.5, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + 1.8, y, "%d%%" % round(v), ha="left", va="center",
                fontsize=FS_VALUE, fontweight="bold", color=VALUE, zorder=4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=INK, labelsize=FS_TICK, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Utilization (%)", fontsize=FS_LABEL, color=INK, labelpad=2)
    # header strip in figure coordinates, directly above the panel
    x0, y0, w, h = ax.get_position().bounds
    fig.patches.append(Rectangle((x0, y0 + h), w, 0.09, transform=fig.transFigure,
                                 facecolor=HEADER, edgecolor="none", zorder=0))
    fig.text(x0 + 0.012, y0 + h + 0.045, "Utilization", ha="left", va="center",
             fontsize=FS_HEAD, fontweight="bold", color="black")
    fig.text(x0 + w - 0.012, y0 + h + 0.045, title, ha="right", va="center",
             fontsize=FS_HEAD, fontweight="bold", color=INK)
    return ys


ys = panel(axL, lut_sh, "LUT")
panel(axD, dsp_sh, "DSP")
axL.set_yticks(ys)
axL.set_yticklabels(names, fontsize=FS_TICK, color=INK)
axD.set_yticks([])

save(fig, "fig11_engine_res_2.0")
