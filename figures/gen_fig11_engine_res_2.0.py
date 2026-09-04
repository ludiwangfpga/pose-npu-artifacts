# -*- coding: utf-8 -*-
"""
Figure 8 of the article: how the nine engine classes divide the two scarce
resources, LUT and DSP.

Two small multiples share one percentage axis, and both denominators are the
nine-engine totals, so the panels are directly comparable: the fused attention
engine takes a quarter of the LUTs and a tenth of the DSPs, the 3x3 convolution
the reverse. The underlying counts are in Table 2 and in
results/engines_post_route.csv.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fig_style import *          # noqa: F401,F403
import matplotlib.pyplot as plt

use_style()

MUL = "×"

# --------------------------------------------------------------- data ---
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
names = [e[0] for e in ENGINES]
lut_sh = [100.0 * e[1] / LUT_TOT for e in ENGINES]
dsp_sh = [100.0 * e[2] / DSP_TOT for e in ENGINES]

DEV_LUT, DEV_DSP = 277400.0, 2020.0
EX_LUT, EX_DSP = 41529.0, 704.0
BUILT = [("Slice", 78.84), ("LUT", 56.48), ("DSP", 42.03)]
EXTRA = {"Slice": 0.0, "LUT": EX_LUT / DEV_LUT * 100.0,
         "DSP": EX_DSP / DEV_DSP * 100.0}

# ------------------------------------------------------------- colours ---
ACC = ORANGE_C3          # the fused attention engine (contribution 3)
DIM = "#a9a9a9"          # de-emphasis grey for the other engines
FILL = C_MID             # as-built fill in the meter
TRACK = C_LIGHT          # the unfilled part of the meter track
RULE = C_GRAY            # hairline axes
DASH = (0, (2.6, 1.6))   # projection, and nothing else, is dashed

# --------------------------------------------------------------- figure ---
fig = plt.figure(figsize=(COL1 + 0.6, 2.30))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                      wspace=0.16, left=0.245, right=0.975,
                      top=0.90, bottom=0.235)
axL = fig.add_subplot(gs[0, 0])
axD = fig.add_subplot(gs[0, 1], sharey=axL)

n = len(ENGINES)
ypos = [n - 1 - i for i in range(n)]
BH = 0.56
XMAX = 72.0


def multiple(ax, vals, unit, label_rows):
    for i, y in enumerate(ypos):
        c = ACC if i == 0 else DIM
        if vals[i] > 0:
            ax.barh(y, vals[i], height=BH, color=c, edgecolor="none", zorder=3)
        else:
            ax.text(1.2, y, "0", ha="left", va="center", fontsize=FS_NOTE,
                    color=INK, zorder=4)
    for i in label_rows:
        ax.text(vals[i] + 1.4, ypos[i], "%.0f" % vals[i], ha="left",
                va="center", fontsize=FS_NOTE, color=INK, zorder=4)
    ax.set_xlim(0, XMAX)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel(unit, fontsize=FS_LABEL, labelpad=2)
    strip(ax, left=False, bottom=True)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_bounds(0, 70)
    ax.tick_params(axis="x", color=RULE)
    ax.tick_params(axis="y", length=0, pad=4)


multiple(axL, lut_sh, "% of engine LUT", (0, 1))
multiple(axD, dsp_sh, "% of engine DSP", (0, 1))
axL.set_ylim(-0.75, n - 1 + 0.75)
axL.set_yticks(ypos)
axL.set_yticklabels(names, fontsize=FS_TICK)
plt.setp(axD.get_yticklabels(), visible=False)

# identity of the emphasised row comes from the accent bar that sits right
# beside its label -- no swatch, and the label itself stays in ink.
# one title over BOTH multiples, so they read as one panel with two measures
_x0 = axL.get_position().x0
_x1 = axD.get_position().x1


save(fig, "fig11_engine_res_2.0")
