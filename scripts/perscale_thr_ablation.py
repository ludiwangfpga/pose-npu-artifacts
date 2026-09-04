# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Ablation behind Table 4 and Table 5 of the article: the probability
# threshold is compiled into one integer constant per scale, and the
# cross-scale ranking is compared in the unified domain against the raw
# code domain. It needs the candidate set, which is not part of this
# release; its output is results/perscale_thr_ablation.json.
# The comments below are the authors' working notes, in Chinese.
"""贡献二的消融：逐尺度阈值编译 + 跨尺度统一域排序。

要证的两件事（对照物 = TRETS 2024 §6.2 的单常数版本）
--------------------------------------------------
A 判定：单个整数常量 n_T 作用于 raw q   vs   逐尺度常量 n_{T,k}
B 排序：K_cap 兜底时按 raw q 取 top-K   vs   按统一 logit 域 q*s_k 取 top-K

三个尺度的 cls 末层量化 scale 不同（P3 .10503 / P4 .22544 / P5 .34163），
所以同一个 raw 值在不同尺度代表不同概率：P4 的 -21 与 P5 的 -13 才是同一个 p=0.01。
单常量/raw 排序都会系统性偏袒 scale 小的尺度。

方法
----
复用 thr_oracle 的 val 落盘（sq640，8400 格概率 + top-300 候选，几何自洽）。
候选是 one2one 按分数降序的 top-300，所以"第 j 个候选"就对应"第 j 大的格子"，
据此把候选映射回格号 -> 尺度。再按各策略量化、判定/排序，跑 pycocotools。

量化仿真：q_k = clamp(round(logit(p)/s_k), -128, 127)，与硬件持有的数一致（含钳位）。
用 venv312。
"""
import os, sys, io, json, glob, argparse, contextlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = r"<project root>"
DIR = os.path.join(ROOT, "toolchain", "repr_sweep", "thr_oracle")
ANN = r"<project root>/..\datasets\coco-pose\annotations\person_keypoints_val2017.json"
CALIB = os.path.join(ROOT, "toolchain", "repr_sweep",
                     os.environ.get("POSE_CALIB", "<calibration-table>.pt"))
OUT = os.path.join(ROOT, "toolchain", "repr_sweep", "perscale_thr_ablation.json")

SPLIT = [(0, 6400, "P3"), (6400, 8000, "P4"), (8000, 8400, "P5")]
T_TARGET = 0.01
KCAPS = [10, 20, 40, 100]
NBINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 999, ">=9")]


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    import torch
    sc_all = torch.load(CALIB, map_location="cpu", weights_only=False)["scales"]
    S_K = np.array([sc_all["H0.cls.2"][0], sc_all["H1.cls.2"][0],
                    sc_all["H2.cls.2"][0]], np.float64)
    print("cls 末层 scale:  " + "  ".join(
        f"{t}={v:.5f}" for (_, _, t), v in zip(SPLIT, S_K)))

    meta = json.load(io.open(os.path.join(DIR, "val_meta.json"), encoding="utf-8"))
    sh = sorted(glob.glob(os.path.join(DIR, "val_shard*.npz")))
    P = np.concatenate([np.load(f)["scores"] for f in sh]).astype(np.float64)
    C = np.concatenate([np.load(f)["cands"] for f in sh]).astype(np.float64)
    N = len(meta)
    iids = np.array([q["image_id"] for q in meta])
    npers = np.array([q["n_person"] for q in meta])

    # 候选 j <-> 第 j 大的格子。先验证这个对应关系成立。
    order = np.argsort(-P, axis=1)[:, :300]
    ref = np.take_along_axis(P, order, axis=1)
    live = C[:, :, 0] > 0
    err = np.abs(C[:, :, 0] - ref)[live].max()
    print(f"候选<->格子 对应校验：最大分数偏差 {err:.2e}  "
          f"({'通过' if err < 2e-3 else '失败，勿采信下表'})")

    # 每个候选的尺度号与量化 q
    cell = order                                   # (N,300) 格号
    scale_id = np.zeros_like(cell)
    for si, (lo, hi, _) in enumerate(SPLIT):
        scale_id[(cell >= lo) & (cell < hi)] = si
    s_of = S_K[scale_id]                           # (N,300)
    q = np.clip(np.round(logit(ref) / s_of), -128, 127)     # int8 域
    q_uni = q * s_of                               # 统一 logit 域
    frac = [float((scale_id[live] == i).mean()) for i in range(3)]
    print("候选的尺度构成：" + "  ".join(
        f"{t} {f:.1%}" for (_, _, t), f in zip(SPLIT, frac)))

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(ANN)
    groups = [(t, set(iids[(npers >= lo) & (npers <= hi)].tolist()))
              for lo, hi, t in NBINS]

    def ev(res, ids):
        if not res:
            return 0.0
        e = COCOeval(coco, coco.loadRes(res), "keypoints")
        e.params.imgIds = list(ids)
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate(); e.accumulate(); e.summarize()
        return float(e.stats[0])

    def build(keep):
        """(N,300) bool -> COCO det 列表。向量化取下标，避免逐格 python 循环。"""
        ii, jj = np.nonzero(keep & live)
        kp = C[ii, jj, 1:]
        sc = C[ii, jj, 0]
        im = iids[ii]
        return [dict(image_id=int(a), category_id=1, score=float(b),
                     keypoints=c.tolist())
                for a, b, c in zip(im, sc, kp)]

    def score(keep, per_bin=True):
        """per_bin=False 时只跑一次 COCOeval（扫描用，快 6 倍）。"""
        res = build(keep)
        tot = ev(res, iids.tolist())
        if not per_bin:
            return tot, None, len(res)
        per = [ev([r for r in res if r["image_id"] in g], g) for _, g in groups]
        return tot, per, len(res)

    tot0, per0, _ = score(live.copy())
    print(f"\n基线（全候选 conf>=0.001）总 AP {tot0:.4f}   " +
          " ".join(f"{t}:{v:.4f}" for (t, _), v in zip(groups, per0)))

    rows = {}

    def report(tag, keep):
        tot, per, n = score(keep)
        rows[tag] = dict(AP=round(tot, 4), dAP=round(tot - tot0, 4),
                         per_bin={t: round(v, 4) for (t, _), v in zip(groups, per)},
                         d_bin={t: round(v - b, 4)
                                for (t, _), v, b in zip(groups, per, per0)},
                         n_sent=n)
        print(f"{tag:<26}{tot:>8.4f}{tot-tot0:>+9.4f}{n:>9d}   " +
              " ".join(f"{v-b:+.4f}" for v, b in zip(per, per0)))

    # ---------- A 判定：逐尺度常量 vs 最优单常量 ----------
    nTk = np.floor(logit(T_TARGET) / S_K).astype(int)   # 往安全侧（更负）取整
    print(f"\nT={T_TARGET} 编译成逐尺度常量: " +
          "  ".join(f"{t}={n}" for (_, _, t), n in zip(SPLIT, nTk)) +
          "   (实际生效 p: " + ", ".join(
              f"{1/(1+np.exp(-n*s)):.4f}" for n, s in zip(nTk, S_K)) + ")")
    print(f"\n{'策略':<26}{'总AP':>8}{'ΔAP':>9}{'发送数':>9}   逐箱 ΔAP "
          f"({' '.join(t for t,_ in groups)})")

    keep_ps = q >= nTk[scale_id]
    report("A 逐尺度常量(本方案)", keep_ps)

    # 原式（TRETS §6.2）：给定同一个 T，逆 sigmoid 映射成【一个】常数，直接用。
    # 它没有搜索步骤——之前我扫 -50..-3 挑最优是给它加了原论文没有的调优，
    # 而且拿 val 的 AP 当目标函数挑（在考卷上调参），两头都不对，已删。
    # 单常数只能按某一个尺度的 s 编译，三种选法各跑一次，照实报。
    for si, (_, _, tag) in enumerate(SPLIT):
        n1 = int(np.floor(logit(T_TARGET) / S_K[si]))
        eff = [1 / (1 + np.exp(-n1 * sk)) for sk in S_K]
        key = f"A 单常量(按{tag}编) n={n1}"        # report 用同一个 key 建行
        report(key, q >= n1)
        rows[key]["n_T"] = n1
        rows[key]["eff_p"] = [round(v, 5) for v in eff]
        print(f"{'':26}  -> 各尺度实际生效 p: " +
              ", ".join(f"{t}={v:.5f}" for (_, _, t), v in zip(SPLIT, eff)))

    # ---------- B 排序：K_cap 兜底 ----------
    print()
    for K in KCAPS:
        for dom, val in (("统一域", q_uni), ("raw域", q)):
            v = np.where(live, val, -1e9)
            idx = np.argsort(-v, axis=1)[:, :K]
            keep = np.zeros_like(live)
            np.put_along_axis(keep, idx, True, axis=1)
            report(f"B top-{K} {dom}", keep & live)

    json.dump(dict(scales=S_K.tolist(), nTk=nTk.tolist(), T=T_TARGET,
                   base_AP=tot0, rows=rows),
              io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
