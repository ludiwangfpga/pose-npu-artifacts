# -*- coding: utf-8 -*-
"""T4-ddr-map-readback verification: every number in docs/ddr_map_pose.md is
computed here from primary sources (instruction streams, blob serializers,
checkpoint shapes, ISA encoder range checks, protocol constants from RTL)."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"<project root>\ultralytics")
sys.path.insert(0, r"<project root>\toolchain")

import numpy as np
from npu_isa import V3, ENGINES, TYPES, read_hex, decode_stream, encode_stream, MAX_STREAM_WORDS

TC = r"<project root>\toolchain"
POOL_LO, POOL_HI = 0x11000000, 0x27000000
SLOT_NR = 0x01000000
WSLOT = 0x00100000
IMG_ADDR = 0x10000000
DDR_TOP = 0x40000000

print("=" * 78)
print("[1] instruction streams: word counts, per-type histogram, words-per-instr")
print("=" * 78)
wpi = {t: 1 + len(spec) for t, spec in V3.items()}  # header + payload
print("words/instr (V3):", {ENGINES[t]: n for t, n in sorted(wpi.items())})

streams = {}
for name, path in [("deploy(unfused)", f"{TC}/export_p3_deploy/p3_instruction_mem.txt"),
                   ("fused", f"{TC}/export_p3_fused/p3_instruction_mem.txt")]:
    words = read_hex(path)
    blocks = decode_stream(words, V3)
    hist = {}
    for t, _ in blocks:
        hist[ENGINES[t]] = hist.get(ENGINES[t], 0) + 1
    streams[name] = (words, blocks)
    print(f"{name}: {len(words)} words, {len(blocks)} instructions, hist={hist}")
    # word accounting closes exactly
    assert sum(wpi[t] for t, _ in blocks) == len(words)
    print(f"  word ledger closes: sum(words/instr) == {len(words)} OK")

print()
print("=" * 78)
print("[2] deploy stream: feature-pool slot usage + head/img/weight address audit")
print("=" * 78)
words, blocks = streams["deploy(unfused)"]
pool_slots, head_addrs, waddrs = set(), set(), set()
for t, f in blocks:
    for k, v in f.items():
        if not k.endswith("addr"):
            continue
        assert v % 8 == 0, f"{ENGINES[t]}.{k}=0x{v:X} not 8B aligned"
        if POOL_LO <= v < POOL_HI:
            pool_slots.add((v - POOL_LO) // SLOT_NR)
        elif v >= POOL_HI:
            head_addrs.add(v)
        elif v < IMG_ADDR:
            waddrs.add(v)
pool_top = POOL_LO + (max(pool_slots) + 1) * SLOT_NR
print(f"pool slots used: {sorted(pool_slots)}  (n={len(pool_slots)})")
print(f"pool top: 0x{pool_top:X}  ({(pool_top-POOL_LO)//SLOT_NR} slots x 16 MiB)")
print(f"head addrs in stream: {sorted(hex(a) for a in head_addrs)}")
print(f"weight-arena addrs: {len(waddrs)} distinct, max 0x{max(waddrs):X}")

print()
print("=" * 78)
CKPT = os.environ.get("POSE_CKPT", "<checkpoint>.pt")
print("[3] kpt tower shapes from %s (CPU)" % CKPT)
print("=" * 78)
import torch
ck = torch.load(f"{TC}/{CKPT}", map_location="cpu", weights_only=False)
mm = ck["ema"] if ck.get("model") is None else ck["model"]   # gate ckpt keeps the EMA model
# find the Pose head (module 23)
head = None
for mod in mm.modules():
    if mod.__class__.__name__ in ("Pose26", "Pose"):
        head = mod
        break
assert head is not None, "Pose head not found"
shapes = {}
for i in range(3):
    c0 = head.one2one_cv4[i][0].conv.weight.shape  # [85, cin, 3, 3]
    c1 = head.one2one_cv4[i][1].conv.weight.shape  # [85, 85, 3, 3]
    c2 = head.one2one_cv4_kpts[i].weight.shape     # [51, 85, 1, 1]
    shapes[i] = (tuple(c0), tuple(c1), tuple(c2))
    print(f"H{i}: cv4[0] {tuple(c0)}  cv4[1] {tuple(c1)}  cv4_kpts {tuple(c2)}")
CIN = {0: shapes[0][0][1], 1: shapes[1][0][1], 2: shapes[2][0][1]}
CMID = shapes[0][0][0]; CKPT = shapes[0][2][0]
assert (CMID, CKPT) == (85, 51) and CIN == {0: 64, 1: 128, 2: 256}
print(f"confirmed: cin={CIN}, mid={CMID}->pad96, kpt={CKPT}->pad64")
print("nc =", head.nc, " kpt_shape =", getattr(head, "kpt_shape", None))

print()
print("=" * 78)
print("[4] kpt weight blob sizes -- via the REAL serializers on zero arrays")
print("=" * 78)
from export_p1 import conv33_blob, conv11_blob


def sz33(cin, cout):
    cin_p = (cin + 15) // 16 * 16; cout_p = (cout + 15) // 16 * 16
    b = conv33_blob(np.zeros((9, cout_p, cin_p), np.int8), np.zeros(cout_p, np.int64))
    assert len(b) == 9 * cout_p * cin_p + 16 * cout_p
    return len(b), cin_p, cout_p


def sz11(cin, cout):
    cin_p = (cin + 15) // 16 * 16; cout_p = (cout + 15) // 16 * 16
    b = conv11_blob(np.zeros((cout_p, cin_p), np.int8), np.zeros(cout_p, np.int64))
    assert len(b) == cout_p * cin_p + 8 * cout_p
    return len(b), cin_p, cout_p


tot_w = 0
blob_rows = []
for i, (H, W) in enumerate([(80, 80), (40, 40), (20, 20)]):
    b0, _, _ = sz33(CIN[i], CMID)
    b1, _, _ = sz33(CMID, CMID)
    b2, _, _ = sz11(CMID, CKPT)
    blob_rows.append((i, b0, b1, b2))
    tot_w += b0 + b1 + b2
    print(f"H{i}.kpt.0 conv33 {CIN[i]}->96: {b0:7,d} B   "
          f"kpt.1 conv33 96->96: {b1:6,d} B   kpt.2 conv11 96->64: {b2:5,d} B")
print(f"kpt weight total: {tot_w:,d} B = {tot_w/1024:.1f} KiB -> 9 blobs x 1 MiB slot = 9 MiB arena increment")

print()
print("=" * 78)
print("[5] encode 9 kpt instructions with real geometry -> V3 field range check")
print("=" * 78)
kins = []
wa = 114 * WSLOT
for i, (Hs, Ws) in enumerate([(80, 80), (40, 40), (20, 20)]):
    for j, (cin, cout, op) in enumerate([(CIN[i], CMID, "conv33"), (CMID, CMID, "conv33"), (CMID, CKPT, "conv11")]):
        cin_p = (cin + 15) // 16 * 16; cout_p = (cout + 15) // 16 * 16
        cout_grp = cout_p // 8 if op == "conv33" else cout_p // 16
        blen = (9 * cout_p * cin_p + 16 * cout_p) if op == "conv33" else (cout_p * cin_p + 8 * cout_p)
        wnum = (cout_p // 8) * (cin_p // 16) * 8 if op == "conv33" else (cin_p // 16) * (cout_p // 16)
        f = dict(s_addr=0x11000000, m_addr=0x12000000, weight_addr=wa,
                 col_channel=Ws * cin_p, m_data_len=Ws * cout_p, col_num=Ws,
                 calc_cin_num=cin_p // 16, calc_cout_num=cout_grp,
                 calc_num=(cin_p // 16) * cout_grp, calc_row_num=Hs,
                 channel_in=cin_p, channel_out=cout_p, weight_len=blen, weight_num=wnum,
                 scale=0x7FFF, zero3=0)
        f.update(dict(zero1=0, stride2=0) if op == "conv33" else dict(relu=int(j < 2)))
        kins.append((TYPES[op], f))
kwords = encode_stream(kins, V3)
print(f"9 kpt instructions encode OK (all fields in range): {len(kwords)} words")
assert len(kwords) == 6 * wpi[0] + 3 * wpi[1]

print()
print("=" * 78)
print("[6] head buffers, readback bytes, packets, time model")
print("=" * 78)
# One 0x08 read = ONE Ethernet frame: ip_tx.v sets total len = udp_len+20, flags
# fixed 0x4000 (DF), zero fragmentation logic. So the practical chunk is
# min(RTL cap 2047, what the host NIC accepts):
#   standard MTU 1500 -> UDP payload <= 1472 -> chunk 1472 (already 8-aligned)
#   jumbo frames on   -> RTL cap 2047   -> chunk 2040 (largest 8-multiple)
OVH = 8 + 14 + 20 + 8 + 4 + 12   # preamble+SFD, Eth, IP, UDP, FCS, IFG = 66 B/frame on wire
MINFRAME = 84                    # 64B min frame + preamble/SFD 8 + IFG 12
NSB = 8e-9                       # 1 Gbps = 8 ns/byte


def region(H, W, C):
    return H * W * C


bufs = {
    "box": [region(80, 80, 16), region(40, 40, 16), region(20, 20, 16)],
    "cls(nc=1 pad16)": [region(80, 80, 16), region(40, 40, 16), region(20, 20, 16)],
    "kpt(51 pad64)": [region(80, 80, 64), region(40, 40, 64), region(20, 20, 64)],
}


def t_ms(nbytes, npk, rtt_us):
    t_data = (nbytes + npk * OVH) * NSB
    t_reqack = npk * 2 * MINFRAME * NSB
    t_rtt = npk * rtt_us * 1e-6
    return (t_data + t_reqack + t_rtt) * 1e3


for CHUNK in (1472, 2040):
    assert CHUNK % 8 == 0 and CHUNK <= 2047

    def pkts(n, c=CHUNK):
        return -(-n // c)

    print()
    print(f"--- chunk = {CHUNK} B/read "
          f"({'standard MTU 1500' if CHUNK == 1472 else 'jumbo frames'}) ---")
    for k, v in bufs.items():
        rems = [x - (pkts(x) - 1) * CHUNK for x in v]
        assert all(r % 8 == 0 for r in rems), f"tail not 8B-aligned: {rems}"
        print(f"{k}: {v}  sum={sum(v):,d} B  pkts={[pkts(x) for x in v]} "
              f"sum={sum(pkts(x) for x in v)}  tails={rems}")
    det_b = sum(bufs["box"]) + sum(bufs["cls(nc=1 pad16)"])
    det_p = sum(pkts(x) for x in bufs["box"]) + sum(pkts(x) for x in bufs["cls(nc=1 pad16)"])
    pose_b = sum(bufs["kpt(51 pad64)"]); pose_p = sum(pkts(x) for x in bufs["kpt(51 pad64)"])
    dual_b, dual_p = det_b + pose_b, det_p + pose_p
    print(f"{'stream':<12}{'bytes':>10}{'pkts':>6} | line-rate floor | RTT=20us | RTT=50us | RTT=100us")
    for nm, b, p in [("detect", det_b, det_p), ("pose", pose_b, pose_p), ("dual", dual_b, dual_p)]:
        print(f"{nm:<12}{b:>10,d}{p:>6d} | {t_ms(b, p, 0):9.2f} ms   | {t_ms(b, p, 20):6.2f} ms"
              f"| {t_ms(b, p, 50):6.2f} ms| {t_ms(b, p, 100):7.2f} ms")

# cross-check: the measured COCO-80 detect deployment reads box16+cls80
coco_b = region(80, 80, 96) + region(40, 40, 96) + region(20, 20, 96)
print(f"\ncross-check: measured COCO-80 deployment readback = box16+cls80 = {coco_b:,d} B "
      f"(== pose dual {dual_b:,d}: {coco_b == dual_b})")

# anchor blob
anc = (80 * 80 + 40 * 40 + 20 * 20) * 4
print(f"anchor const region: 8400 cells x 4 B (u16 Q8.8 ax,ay) = {anc:,d} B "
      f"(16B-mult: {anc % 16 == 0})")

print()
print("=" * 78)
print("[7] instruction budget")
print("=" * 78)
inc = 6 * wpi[0] + 3 * wpi[1]
for nm, (w, blk) in streams.items():
    tot = len(w) + inc
    print(f"{nm}: {len(w)} + {inc} (kpt: 6xconv33@{wpi[0]}w + 3xconv11@{wpi[1]}w) = {tot}"
          f"  margin vs {MAX_STREAM_WORDS}: {MAX_STREAM_WORDS - tot} words"
          f" ({(MAX_STREAM_WORDS - tot) / MAX_STREAM_WORDS * 100:.1f}%)")

print()
print("=" * 78)
print("[8] map-level asserts (pose map)")
print("=" * 78)
arena_top_now = 114 * WSLOT
arena_top_pose = (114 + 9 + 1) * WSLOT     # +9 kpt blobs +1 anchor slot
assert arena_top_pose <= IMG_ADDR
print(f"weight arena: now 0x{arena_top_now:X} (114 MiB) -> pose 0x{arena_top_pose:X} "
      f"({arena_top_pose//WSLOT} MiB) <= IMG 0x{IMG_ADDR:X} OK")
pool_top_worst = pool_top + 2 * SLOT_NR    # worst case: kpt intermediates get 2 fresh slots
assert pool_top_worst <= POOL_HI
print(f"feature pool: now top 0x{pool_top:X} -> worst-case pose 0x{pool_top_worst:X} "
      f"<= 0x{POOL_HI:X} OK (margin {(POOL_HI - pool_top_worst)//SLOT_NR} slots)")
for hb in (0x29000000, 0x28000000, 0x27000000):
    spans = [(hb, region(80, 80, 16)), (hb + 0x100000, region(80, 80, 16)), (hb + 0x200000, region(80, 80, 64))]
    for (a0, s0), (a1, _s1) in zip(spans, spans[1:]):
        assert a0 + s0 <= a1, f"overlap at 0x{a0:X}"
hi_ext = 0x29000000 + 0x300000
assert hi_ext < DDR_TOP
print(f"head regions: box@+0x0, cls@+0x100000, kpt@+0x200000 no overlap; map top 0x{hi_ext:X} < 1 GiB OK")
print("\nALL CHECKS PASSED")
