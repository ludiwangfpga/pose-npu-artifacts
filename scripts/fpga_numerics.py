# -*- coding: utf-8 -*-
"""Bit-accurate Python mirror of the NPU integer arithmetic (from RTL).

Validated against the v5 reference dumps before use as the P1 quant reference.
Each function reproduces the exact fixed-point path of its RTL engine:
  conv33 / conv11 : u8 = clamp( round( (acc*scale + bias) >> 16 )[relu] + zero3 )
  add             : u8 = clamp( round( ((r(x0-z0)*s0 + r(x1-z1)*s1) * s2) >> 32 ) )
  cat             : same as add but >> 30
"""
import numpy as np


def _round_shift(v, sh):
    """>>sh with round-half-up via the top dropped bit (matches RTL pipe[..]+pipe[sh-1])."""
    return (v >> sh) + ((v >> (sh - 1)) & 1)


def conv_requant(acc, scale, bias, zero3, relu):
    """conv33/conv11 output stage. acc: int64 array of raw MAC sums. Returns uint8."""
    t = acc.astype(np.int64) * int(scale) + bias.astype(np.int64)   # + pre-scaled int32 bias
    if relu:
        t = np.where(t < 0, 0, t)
    out = _round_shift(t, 16) + int(zero3)
    return np.clip(out, 0, 255).astype(np.uint8)


def conv33(feat_u8, weight_i8, bias_i32, scale, zero1, zero3, stride, relu=True):
    """3x3 conv, NHWC uint8 in, uint8 out. feat: [H,W,Cin], weight: [9,Cout,Cin]
    tap order row-major (r0c0,r0c1,r0c2,r1c0,...). Pad=1 with zero1."""
    H, W, Cin = feat_u8.shape
    ntap, Cout, _ = weight_i8.shape
    assert ntap == 9
    pad = np.full((H + 2, W + 2, Cin), zero1, dtype=np.int64)
    pad[1:H + 1, 1:W + 1, :] = feat_u8
    Ho, Wo = H // stride, W // stride
    acc = np.zeros((Ho, Wo, Cout), dtype=np.int64)
    fq = pad.astype(np.int64)                     # data treated unsigned
    for t in range(9):
        dr, dc = t // 3, t % 3
        win = fq[dr:dr + Ho * stride:stride, dc:dc + Wo * stride:stride, :]  # [Ho,Wo,Cin]
        acc += np.einsum('hwc,oc->hwo', win, weight_i8[t].astype(np.int64))
    out = conv_requant(acc.reshape(-1, Cout), scale, np.tile(bias_i32, (Ho * Wo, 1)),
                       zero3, relu)
    return out.reshape(Ho, Wo, Cout)


def conv11(feat_u8, weight_i8, bias_i32, scale, zero3, relu=True):
    """1x1 conv. weight: [Cout,Cin]."""
    H, W, Cin = feat_u8.shape
    Cout = weight_i8.shape[0]
    acc = np.einsum('hwc,oc->hwo', feat_u8.astype(np.int64), weight_i8.astype(np.int64))
    out = conv_requant(acc.reshape(-1, Cout), scale, np.tile(bias_i32, (H * W, 1)), zero3, relu)
    return out.reshape(H, W, Cout)


def add(x0_u8, x1_u8, scale0, scale1, scale2, zero0, zero1):
    a = np.clip(x0_u8.astype(np.int64) - zero0, 0, None) * int(scale0)
    b = np.clip(x1_u8.astype(np.int64) - zero1, 0, None) * int(scale1)
    t = (a + b) * int(scale2)
    out = _round_shift(t, 32)
    return np.clip(out, 0, 255).astype(np.uint8)


def cat_requant(x_u8, scale, scale2, zero):
    a = np.clip(x_u8.astype(np.int64) - zero, 0, None) * int(scale)
    t = a * int(scale2)
    return np.clip(_round_shift(t, 30), 0, 255).astype(np.uint8)


def cat(x0_u8, x1_u8, s0, s1, s2, z0, z1):
    """Channel concat with per-source requant, on last axis."""
    o0 = cat_requant(x0_u8, s0, s2, z0)
    o1 = cat_requant(x1_u8, s1, s2, z1)
    return np.concatenate([o0, o1], axis=-1)


# ================= V3 attention ops (P3) =================
# All four mirror their RTL engines bit-for-bit (dw_calc / mm_calc / softmax_calc),
# each already cross-checked 0-mismatch against these exact numpy formulas.

def dwconv(feat_u8, weight_i8, bias_i32, scale, zero1, zero3, relu=False):
    """3x3 stride-1 depthwise conv. feat: [H,W,C] uint8, weight: [9,C] int8
    (tap row-major r0c0..r2c2, per-channel), bias: [C] int32. Pad=1 with zero1."""
    H, W, C = feat_u8.shape
    ntap, Cw = weight_i8.shape
    assert ntap == 9 and Cw == C
    pad = np.full((H + 2, W + 2, C), zero1, dtype=np.int64)
    pad[1:H + 1, 1:W + 1, :] = feat_u8
    acc = np.zeros((H, W, C), dtype=np.int64)
    for t in range(9):
        dr, dc = t // 3, t % 3
        acc += weight_i8[t].astype(np.int64) * pad[dr:dr + H, dc:dc + W, :]
    t = acc * int(scale) + bias_i32.astype(np.int64)[None, None, :]
    if relu:
        t = np.where(t < 0, 0, t)
    out = _round_shift(t, 16) + int(zero3)
    return np.clip(out, 0, 255).astype(np.uint8)


def k_requant(k_u8, k_zero, k_scale):
    """K load-path requant to symmetric int8 (matmul qk mode), s_k2 frozen."""
    return np.clip(_round_shift((k_u8.astype(np.int64) - int(k_zero)) * int(k_scale), 16),
                   -128, 127)


def matmul_qk(q_u8, k_u8, scale, z_q, z_out, k_zero, k_scale):
    """Q @ K^T attention logits. q,k: [N,kd] uint8. Returns [N,N] uint8."""
    k2 = k_requant(k_u8, k_zero, k_scale)
    sumK = k2.sum(axis=1)
    acc = q_u8.astype(np.int64) @ k2.T - int(z_q) * sumK[None, :]
    out = _round_shift(acc * int(scale), 16) + int(z_out)
    return np.clip(out, 0, 255).astype(np.uint8)


def softmax_lut(s_in, bits=16):
    """256-entry exp LUT: LUT[i] = round(exp((i-255)*s_in) * (2^bits-1)), LUT[255]=full."""
    i = np.arange(256, dtype=np.float64)
    lut = np.round(np.exp((i - 255.0) * float(s_in)) * ((1 << bits) - 1))
    lut = np.clip(lut, 0, (1 << bits) - 1).astype(np.int64)
    lut[255] = (1 << bits) - 1
    return lut


def softmax_rows(x_u8, s_in):
    """Row softmax via exp LUT + reciprocal multiply (matches softmax_calc).
    x: [rows,cols] uint8. Returns uint8, output scale=1/255 zero=0."""
    lut = softmax_lut(s_in)
    x = x_u8.astype(np.int64)
    row_max = x.max(axis=1, keepdims=True)
    e = lut[x + (255 - row_max)]
    S = e.sum(axis=1, keepdims=True)
    R = (1 << 31) // S
    out = (e * 255 * R) >> 31
    return np.minimum(out, 255).astype(np.uint8)


def matmul_av(p_u8, v_u8, scale, z_v, z_out):
    """attn @ V. p: [N,N] uint8 (softmax out, zero=0), v: [N,hd] uint8. Returns [N,hd] uint8."""
    sumP = p_u8.astype(np.int64).sum(axis=1)
    acc = p_u8.astype(np.int64) @ v_u8.astype(np.int64) - int(z_v) * sumP[:, None]
    out = _round_shift(acc * int(scale), 16) + int(z_out)
    return np.clip(out, 0, 255).astype(np.uint8)
