# -*- coding: utf-8 -*-
"""
Figure 11 of the article: the probability actually in effect on each of the
three classification branches under each threshold-compilation scheme.

A deviation-from-target dot plot: the reference line is the specified
probability T, each marker is one branch under one scheme, and the stem ties
the marker to the reference so the deviation is the length of the stem. The
per-item values are in Table 4 of the article.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from matplotlib.lines import Line2D

from fig_style import *          # noqa: F401,F403

use_style()
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"

# ------------------------------------------------------------------ data ---
T = 0.01
STEP = [0.10503, 0.22544, 0.34163]
BRANCH = ['P3', 'P4', 'P5']
SCHEME = [
    ('on P3',        [0.00974, 4.9e-5,   3.0e-7 ]),
    ('on P4',        [0.09925, 8.713e-3, 7.65e-4]),
    ('on P5',        [0.18688, 4.085e-2, 8.303e-3]),
    ('per-scale',    [0.00974, 8.71e-3,  8.30e-3]),
]
XPOS = [0.0, 1.0, 2.0, 3.35]                 # gap before the per-scale group
OFF = (-0.22, 0.0, 0.22)
MARK = ('o', 's', '^')
ACC = GREEN_C2                                # contribution-2 accent
DIM = C_MID                                   # de-emphasis grey

# ---------------------------------------------------------------- canvas ---
fig = plt.figure(figsize=(COL1, 2.75))
ax = fig.add_axes([0.20, 0.235, 0.775, 0.700])
ax.set_yscale('log')
ax.set_ylim(4e-8, 1.6)
ax.set_xlim(-0.55, 3.95)

# recessive decade grid (allowed on a log axis), no vertical grid
ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(-7, 1)]))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, p: r'$10^{%d}$' % round(__import__('math').log10(v))))
ax.grid(axis='y', color=C_LIGHT, lw=0.7, zorder=0)
ax.set_axisbelow(True)

# wash behind the per-scale group (accent at ~8 % opacity)
ax.axvspan(XPOS[3] - 0.45, XPOS[3] + 0.45, color=ACC, alpha=0.08, lw=0, zorder=0)

# the target: the one dashed element in the plot
ax.axhline(T, color=INK, lw=0.9, dashes=(3.0, 2.0), zorder=2)
ax.text(-0.50, T * 1.35, 'T = 0.01', color=INK, fontsize=FS_NOTE,
        ha='left', va='bottom', zorder=6)

# direction cues at the axis edge, muted

# ---------------------------------------------------------- the dots ---
for i, (name, vals) in enumerate(SCHEME):
    this = (i == 3)
    col = ACC if this else DIM
    for k, v in enumerate(vals):
        x = XPOS[i] + OFF[k]
        ax.plot([x, x], [T, v], color=col, lw=0.9, solid_capstyle='butt',
                zorder=3)
        ax.plot([x], [v], marker=MARK[k], color=col, ms=5.2 if this else 4.6,
                mec='white', mew=0.8, ls='none', zorder=5)

# direct labels at the two extremes only (consequence, not the value --
# the value is readable off the axis)
x_lo = XPOS[0] + OFF[2]; y_lo = SCHEME[0][1][2]
x_hi = XPOS[2] + OFF[0]; y_hi = SCHEME[2][1][0]

# ------------------------------------------------------------- legend ---
handles = [Line2D([], [], marker=MARK[k], color=DIM, ls='none', ms=4.6,
                  mec='white', mew=0.8,
                  label=BRANCH[k])
           for k in range(3)]
ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.005),
          ncol=3, frameon=False, fontsize=FS_NOTE, handletextpad=0.4,
          columnspacing=1.3, borderpad=0.0, borderaxespad=0.0)

# ---------------------------------------------------------------- axes ---
strip(ax, left=True, bottom=False)
ax.spines['left'].set_color(C_GRAY)
ax.tick_params(axis='y', color=C_GRAY)
ax.set_ylabel('effective enforced probability', fontsize=FS_LABEL, labelpad=3)

ax.set_xticks(XPOS)
ax.set_xticklabels(['on P3', 'on P4', 'on P5', 'per-scale\n(this work)'],
                   fontsize=FS_TICK)
ax.tick_params(axis='x', length=0, pad=4)
for lbl in ax.get_xticklabels():
    if 'this work' in lbl.get_text():
        lbl.set_fontweight('bold')

# group header under the three shared-constant schemes
ax.plot([-0.40, 2.40], [-0.215, -0.215], color=C_GRAY, lw=0.7,
        transform=ax.get_xaxis_transform(), clip_on=False)
ax.text(1.0, -0.245, 'single shared constant', color='#3d3d3d',
        fontsize=FS_NOTE, ha='center', va='top',
        transform=ax.get_xaxis_transform())

save(fig, 'fig8_thr_compile')
