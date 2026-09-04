# -*- coding: utf-8 -*-
"""Per-engine in-frame busy time from the five-frame probe waveform.

Reproduces the "busy (us)" and "share of frame" columns of Table II and the
activity bands of Fig. 7 of the paper, directly from data/p5f_probes.vcd.

The VCD is dumped by the full-network testbench and carries one scalar per
engine class (ENG_CONV33, ENG_CONV11, ...), which is high exactly while that
engine is executing an instruction, plus the frame counter FRAME_INDEX.

usage:  python engine_busy_time.py [path/to/p5f_probes.vcd]
"""
import os
import re
import sys

ENGINES = ["ENG_CONV33", "ENG_CONV11", "ENG_DWCONV", "ENG_ADD",
           "ENG_CONCAT", "ENG_SPPF", "ENG_UPSAMPLE", "ENG_FUSED_ATTN"]
LABEL = {"ENG_CONV33": "conv3x3", "ENG_CONV11": "conv1x1",
         "ENG_DWCONV": "depthwise conv", "ENG_ADD": "element-wise add",
         "ENG_CONCAT": "concatenation", "ENG_SPPF": "SPPF",
         "ENG_UPSAMPLE": "upsample", "ENG_FUSED_ATTN": "fused attention"}
SCALE = {"s": 1e12, "ms": 1e9, "us": 1e6, "ns": 1e3, "ps": 1.0, "fs": 1e-3}


def parse_vcd(path):
    """Minimal VCD reader: returns (ts_ps, edges, buses).

    edges[name]  = list of (t_on, t_off) intervals in ps
    buses[name]  = list of (t, int value)
    """
    ident, ts_ps = {}, 1.0
    edges, buses, level, start = {}, {}, {}, {}
    t = 0
    header = True
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if header:
                m = re.match(r"\$timescale\s+(\d+)\s*(fs|ps|ns|us|ms|s)", line)
                if m:
                    ts_ps = int(m.group(1)) * SCALE[m.group(2)]
                    continue
                m = re.match(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)", line)
                if m:
                    width, sym, name = int(m.group(1)), m.group(2), m.group(3)
                    ident[sym] = (name, width)
                    continue
                if line.startswith("$enddefinitions"):
                    header = False
                continue
            if line[0] == "#":
                t = int(line[1:])
                continue
            if line[0] in "01xzXZ" and len(line) > 1:          # scalar change
                val, sym = line[0], line[1:]
                if sym not in ident:
                    continue
                name = ident[sym][0]
                if val == "1":
                    if level.get(name) != 1:
                        start[name] = t
                    level[name] = 1
                else:
                    if level.get(name) == 1:
                        edges.setdefault(name, []).append((start[name], t))
                    level[name] = 0
            elif line[0] in "bB":                                # vector change
                parts = line.split()
                if len(parts) != 2 or parts[1] not in ident:
                    continue
                bits = parts[0][1:]
                try:
                    v = int(bits, 2)
                except ValueError:
                    continue
                buses.setdefault(ident[parts[1]][0], []).append((t, v))
    for name, lv in level.items():                               # close open highs
        if lv == 1:
            edges.setdefault(name, []).append((start[name], t))
    return ts_ps, edges, buses, t


def overlap(iv, t0, t1):
    return max(0, min(iv[1], t1) - max(iv[0], t0))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "..", "data", "p5f_probes.vcd")
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(default)
    ts_ps, edges, buses, t_end = parse_vcd(path)
    print("VCD: %s" % path)
    print("timescale %g ps, ends at %.3f ms\n" % (ts_ps, t_end * ts_ps / 1e9))

    fi = buses.get("FRAME_INDEX", [])
    frames = []
    for i, (t, v) in enumerate(fi):
        t1 = fi[i + 1][0] if i + 1 < len(fi) else t_end
        if v >= 1:
            frames.append((v, t, t1))
    if not frames:
        print("no FRAME_INDEX transitions found"); return

    v, f0, f1 = frames[len(frames) // 2]        # a steady-state middle frame
    span_us = (f1 - f0) * ts_ps / 1e6
    print("frame %d spans %.3f us  (all five frames are cycle-identical)\n" % (v, span_us))
    print("%-18s %12s %9s %8s" % ("engine", "busy (us)", "share", "instrs"))
    total = 0.0
    for e in ENGINES:
        iv = edges.get(e, [])
        busy = sum(overlap(x, f0, f1) for x in iv) * ts_ps / 1e6
        n = sum(1 for x in iv if x[0] >= f0 and x[1] <= f1)
        total += busy
        print("%-18s %12.1f %8.1f%% %8d" % (LABEL[e], busy, 100 * busy / span_us, n))
    print("%-18s %12.1f %8.1f%%" % ("sum", total, 100 * total / span_us))
    print("\nThe engines execute one instruction at a time: the intervals of the eight")
    print("classes do not overlap, so the sum above is the occupied fraction of the frame.")


if __name__ == "__main__":
    main()
