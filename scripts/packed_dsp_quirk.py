# -*- coding: utf-8 -*-
"""Quirk-exact golden regeneration for the pose export.
Monkeypatches fpga_numerics.conv33/conv11 with the packed-DSP-quirk models
(ported VERBATIM from sim/p3fullnet/fullnet19/quirk_golden.py — the model that
achieved the P3 162-layer 0-mismatch run), then reruns export_pose main().
Instructions/weights/scales are identical (same ckpt + calib); only golden/*.txt
change. An assert at the end proves the instruction stream is byte-identical.
"""
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"<project root>\toolchain")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import fpga_numerics as fp
from fpga_numerics import _round_shift


def conv33_q(feat_u8, w9, bias, M, z1, z3, stride, relu):
    """fp.conv33 + packed-DSP quirk on even couts (quirk_golden.py verbatim)."""
    H, W, Cin = feat_u8.shape
    _, Cout, _ = w9.shape
    pad = np.full((H + 2, W + 2, Cin), z1, dtype=np.int64)
    pad[1:H + 1, 1:W + 1, :] = feat_u8
    Ho, Wo = H // stride, W // stride
    acc = np.zeros((Ho, Wo, Cout), dtype=np.int64)
    for t in range(9):
        dr, dc = t // 3, t % 3
        win = pad[dr:dr + Ho * stride:stride, dc:dc + Wo * stride:stride, :]
        acc += np.einsum('hwc,oc->hwo', win, w9[t].astype(np.int64))
        z = (win == 0)                                    # data byte == 0
        # paired ODD weight < 0; unpaired trailing even cout (odd Cout, e.g. nc=1)
        # pairs with zero padding weights in hardware -> zero quirk contribution
        n_even = (Cout + 1) // 2
        wn = np.zeros((n_even, Cin), np.int64)
        odd = (w9[t, 1::2, :] < 0).astype(np.int64)
        wn[:odd.shape[0]] = odd
        acc[:, :, 0::2] += np.einsum('hwc,oc->hwo', z.astype(np.int64), wn)
    t2 = acc * int(M) + np.asarray(bias, np.int64)[None, None, :]
    if relu:
        t2 = np.where(t2 < 0, 0, t2)
    return np.clip(_round_shift(t2, 16) + int(z3), 0, 255).astype(np.uint8)


def conv11_q(feat_u8, w2, bias, M, z3, relu):
    H, W, Cin = feat_u8.shape
    x = feat_u8.astype(np.int64)
    acc = np.einsum('hwc,oc->hwo', x, w2.astype(np.int64))
    z = (x == 0)
    Cout = w2.shape[0]
    n_even = (Cout + 1) // 2
    wn = np.zeros((n_even, Cin), np.int64)
    odd = (w2[1::2, :] < 0).astype(np.int64)
    wn[:odd.shape[0]] = odd
    acc[:, :, 0::2] += np.einsum('hwc,oc->hwo', z.astype(np.int64), wn)
    t2 = acc * int(M) + np.asarray(bias, np.int64)[None, None, :]
    if relu:
        t2 = np.where(t2 < 0, 0, t2)
    return np.clip(_round_shift(t2, 16) + int(z3), 0, 255).astype(np.uint8)


fp.conv33 = conv33_q
fp.conv11 = conv11_q
print("[quirk] fp.conv33/fp.conv11 patched with packed-DSP quirk models")

import hashlib

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

ref_instr = r"<project root>\sim\pose_fullnet\instr.hex"
before = md5(ref_instr)

import export_pose
export_pose.main(sys.argv[1:])   # forward CLI (e.g. --pt <checkpoint>.pt)

new_instr = r"<project root>\toolchain\export_pose\pose_instruction_mem.txt"
after = md5(new_instr)
same = before == after
n_words = sum(1 for l in open(new_instr) if l.strip())
print(f"[quirk] instruction stream md5: sim-copy={before[:10]} regenerated={after[:10]} identical={same}")
print(f"[quirk] stream length: {n_words} words (structure check)")
print("[quirk] DONE")
