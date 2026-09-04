# -*- coding: utf-8 -*-
"""
Figure 9 of the article: the AP, AP50 and AP75 that the deployed arm loses
relative to the same-pipeline float32 reference.

The figure plots the LOSS rather than the level, so each bar hangs from the
float32 line and its depth is the gap the text discusses; the absolute levels
are in Table 3. Both arms are evaluated through one pipeline on the COCO
val2017 subset with keypoint annotations (2,346 images, square letterbox 640,
pad 114, score threshold 1e-3, maxDets 20), and the numbers are the ones the
article reports in Table 3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import *          # noqa: F401,F403

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

use_style()

# ------------------------------------------------------------------- data ---
# Units of 1e-4 AP, so the differences below are exact in binary floating
# point: (8007-7841)/100 == 1.66 exactly, which rounds to the "-1.7" printed
# in the body text.  Subtracting the float values directly would not be exact.
METRIC = ['AP', 'AP50', 'AP75']
FLOAT = np.array([5263, 8007, 5689])
FOLD = np.array([4787, 7841, 5087])        # int8_rr8_zfold, the shipped arm

LOSS_FOLD = (FLOAT - FOLD) / 100.0        #  4.76   1.66   6.02

x = np.arange(3.0)
BW = 0.45
OFF = 0.0

fig, ax = plt.subplots(figsize=(COL1, 1.95))

# ----------------------------------------------- bars, hanging from float32 --
# The discarded arm is neutral grey: it de-weights the configuration the paper
# does not ship, and it stays separable from the blue arm in greyscale print
# (183 vs 116 of 255), which C_RED did not.
ax.bar(x + OFF, -LOSS_FOLD, BW, color=BLUE_C1, edgecolor='none', zorder=2)

# --------------------------------------------------- the float32 zero line ---
ax.axhline(0.0, color=C_MID, lw=0.9, zorder=3)

# -------------------------------------------------------- the two numbers ----
# Placed just under the tip of the shipped bar they measure: on a loss axis the
# bar IS the gap, so no dimension stem is needed to bind label to quantity.
for i in (0, 1, 2):
    ax.text(x[i] + OFF, -LOSS_FOLD[i] - 0.25,
            '−%.1f' % LOSS_FOLD[i], ha='center', va='top',
            color=INK, fontsize=FS_NOTE, fontweight='bold', zorder=5)

# ------------------------------------------------------------------- axes ---
YLO = -8.0
ax.set_xlim(-0.58, 2.58)
ax.set_ylim(YLO, 0)
ax.set_yticks([0, -2, -4, -6, -8])
ax.set_yticklabels(['0', '−2', '−4', '−6', '−8'])
ax.set_ylabel('AP lost vs float32 (points)')
ax.set_xticks([])
strip(ax, left=True, bottom=False)
ax.spines['left'].set_bounds(YLO, 0)

# metric names ride above the zero line -- the bars occupy everything below it.
# The line itself needs no "float32" tag: the y-axis label already names the
# reference, and a tag here would sit beside AP75 and read as a fourth metric.
for xi, m in zip(x, METRIC):
    ax.text(xi, 0.15, m, ha='center', va='bottom', color=INK,
            fontsize=FS_TICK, zorder=5)

fig.tight_layout(pad=0.25)
save(fig, 'fig9_quant_chain')
