# -*- coding: utf-8 -*-
"""Pose fullnet judge: compare RTL dumps (d_<name>.txt, decimal per byte) against
toolchain/export_pose/golden/<name>.txt with per-region |d| statistics.

Standard (P3): conv path |d|<=1 normal, rare +-2 recorded verbatim;
attention/fusion path (attn* regions) 0 tolerance; kpt towers judged as conv.
Diff is computed as wrapped int8 distance: ((a-b+128) mod 256) - 128.
"""
import os, re, sys

# Two golden flavors exist (see CLAUDE.md): export_pose.py writes the *ideal* golden,
# export_pose_quirk.py the *packed-DSP quirk* golden. RTL must be judged against the
# QUIRK golden. Override the dir with POSE_GOLDEN to pick a flavor without editing.
GOLD = os.environ.get("POSE_GOLDEN",
                      r"<project root>/toolchain/export_pose/golden")
MAN  = r"<project root>/toolchain/export_pose/manifest.txt"
HERE = os.path.dirname(os.path.abspath(__file__))

ATTN_RE = re.compile(r"attn")   # attention/fusion path: 0 tolerance

names = [l.split()[0] for l in open(MAN) if len(l.split()) == 3]
rows, viol, pend = [], [], []
npass = 0
for i, nm in enumerate(names):
    dpath = os.path.join(HERE, f"d_{nm}.txt")
    gpath = os.path.join(GOLD, f"{nm}.txt")
    if not os.path.exists(dpath):
        pend.append(nm); continue
    d = open(dpath).read().split()
    g = open(gpath).read().split()
    if len(d) != len(g):
        viol.append((i, nm, f"LEN {len(d)} vs {len(g)}")); continue
    if all(x == "0" for x in d[:256]) and not all(x == "0" for x in g[:256]):
        pend.append(nm); continue
    hist = {}
    first = None
    for j, (a, b) in enumerate(zip(d, g)):
        if a == b: continue
        dd = ((int(a) - int(b) + 128) % 256) - 128
        ad = abs(dd)
        hist[ad] = hist.get(ad, 0) + 1
        if first is None: first = (j, a, b)
    maxd = max(hist) if hist else 0
    nmm = sum(hist.values())
    is_attn = bool(ATTN_RE.search(nm))
    limit = 0 if is_attn else 2
    ok = maxd <= limit
    rows.append((i, nm, len(g), nmm, maxd, hist, is_attn, ok, first))
    if ok: npass += 1
    else:  viol.append((i, nm, f"max|d|={maxd} mm={nmm}/{len(g)} first@{first[0]} rtl={first[1]} gold={first[2]}"))

exact = sum(1 for r in rows if r[3] == 0)
print(f"regions compared: {len(rows)}/{len(names)}  PENDING={len(pend)}")
print(f"bit-exact: {exact}   within-standard: {npass}   VIOLATIONS: {len(viol)}")
tot_hist = {}
for r in rows:
    for k, v in r[5].items(): tot_hist[k] = tot_hist.get(k, 0) + v
print("global |d| histogram (nonzero):", dict(sorted(tot_hist.items())) or "{} (all bytes identical)")
nz = [r for r in rows if r[3]]
if nz:
    print("\nregions with any mismatch:")
    for i, nm, n, nmm, maxd, hist, is_attn, ok, first in nz:
        tag = "ATTN" if is_attn else "conv"
        print(f"  #{i:3d} {nm:28s} [{tag}] max|d|={maxd} mm={nmm}/{n} hist={dict(sorted(hist.items()))} {'ok' if ok else 'VIOLATION'}")
if pend:
    print("\npending:", " ".join(pend))
if not viol and not pend and rows:
    print(f"\nVERDICT: PASS (all {len(rows)} regions within P3 standard)")
elif viol:
    print("\nVERDICT: FAIL")
    for v in viol[:20]: print("  VIOL", v)
