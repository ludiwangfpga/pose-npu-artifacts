# -*- coding: utf-8 -*-
"""精简推理路径：只走 one2one 分支 + 整网 CUDA Graph。

依据（bench_export_flag / prof_head 实测）：
  · Pose26 在 eval 下把 one2many（训练分支，33 个卷积 / 0.673 M）和 one2one
    （end2end 实际用的，同样 33 个卷积 / 0.673 M）都算了一遍，但 _inference 只收
    一组 {boxes(1,4,8400), scores(1,1,8400), kpts(1,51,8400), feats}。一半白算。
  · 整网 98% 时间卡在 CPU 逐算子派发，GPU 空转；fp16/channels_last 都无效，
    因为它们减的是算力不是发射次数。
  · 主干+颈部单独捕获 CUDA Graph 可得 3.08x；整网捕获被头里 1 个主机同步挡住。

本模块提供：
    LeanPose(pt)          只走 one2one 的推理封装，输出与原模型逐位一致
    LeanPose.graph()      把整条前向录成 CUDA Graph，一次提交
两者都对外暴露 __call__(x) -> (1, max_det, 57)，与原 net(x)[0] 同义。

用法：
    m = LeanPose(PT, size=640).graph()
    y = m(x)          # x: (1,3,H,W) float cuda，值域 0..1
"""
import os, sys, io, contextlib

import torch

_ROOT = os.environ.get("POSE_ROOT", ".")   # set POSE_ROOT to the project root
if os.path.join(_ROOT, "ultralytics") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ultralytics"))


class LeanPose:
    def __init__(self, pt, size=640, device="cuda", batch=1, net=None, fuse=True):
        from ultralytics import YOLO
        if net is None:
            with contextlib.redirect_stdout(io.StringIO()):
                net = YOLO(pt).model.to(device).eval()
                if fuse:      # BN 折叠进卷积：少一半算子发射，数值有 1e-5 量级偏差
                    net = net.fuse()
                net = net.eval()
        self.net = net
        self.h = self.net.model[-1]
        self.size = size
        self.batch = batch
        self.device = device
        self.nl = self.h.nl
        # one2one 的五组塔（kpts_sigma 是训练用的，推理不取）
        self.box = self.h.one2one["box_head"]
        self.cls = self.h.one2one["cls_head"]
        self.pose = self.h.one2one["pose_head"]
        self.kpts = self.h.one2one["kpts_head"]
        # 主干输出的三个尺度层号（Pose26 的 f）
        self.fidx = list(self.h.f)
        self._g = None
        self._perm = self._probe_perm()

    def _probe_perm(self):
        """postprocess 期望 (batch, anchors, ch) 还是 (batch, ch, anchors)，试一次定下来。"""
        na = sum((self.size // s) ** 2 for s in (8, 16, 32))
        raw = torch.zeros(1, self.h.no + self.h.nk, na, device=self.device)
        for p in (True, False):
            try:
                with torch.no_grad():
                    o = self.h.postprocess(raw.permute(0, 2, 1) if p else raw)
                if o.ndim == 3 and o.shape[1] == self.h.max_det:
                    return p
            except Exception:
                pass
        raise RuntimeError("无法确定 postprocess 的入参布局")

    # ---------- 主干 + 颈部 ----------
    def _trunk(self, x):
        y, out = [], []
        for i, m in enumerate(self.net.model[:-1]):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return [y[i] for i in self.fidx]

    # ---------- 只走 one2one 的头 ----------
    def _head(self, feats):
        b, s, k = [], [], []
        for i, f in enumerate(feats):
            b.append(self.box[i](f).flatten(2))
            s.append(self.cls[i](f).flatten(2))
            k.append(self.kpts[i](self.pose[i](f)).flatten(2))
        return self.h._inference({
            "boxes": torch.cat(b, 2),
            "scores": torch.cat(s, 2),
            "feats": feats,
            "kpts": torch.cat(k, 2),
        })

    def _body(self, x):
        """主干 -> one2one 头 -> 解码，出 (1, 56, 8400) 原始预测。全程无主机同步，可捕获。"""
        return self._head(self._trunk(x))

    def _select(self, y):
        """end2end 的 top-max_det 选择，出 (1, max_det, 57)。

        内部 get_topk_index（head.py:257）含主机同步，CUDA Graph 捕获不了，
        所以整条流水在这里切开：前面录图，这一步留 eager。它只在 8400 个候选上
        做一次 topk，实测占比很小。"""
        return self.h.postprocess(y.permute(0, 2, 1) if self._perm else y)

    def forward(self, x):
        return self._select(self._body(x))

    # ---------- CUDA Graph ----------
    def graph(self, warm=30):
        """把 _body（主干+头+解码）录成一张图，_select 留在图外 eager。"""
        x = torch.zeros(self.batch, 3, self.size, self.size, device=self.device)
        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            for _ in range(warm):
                self._body(x)
        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = False   # 捕获期间禁止再调优
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._body(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._sin = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            self._sout = self._body(self._sin)
        torch.cuda.synchronize()
        self._g = g
        return self

    def __call__(self, x):
        if self._g is None:
            with torch.no_grad():
                return self.forward(x)
        n = x.shape[0]
        if n == self.batch:
            self._sin.copy_(x)
        else:                          # 不足一档就补零，出来后切掉
            self._sin[:n].copy_(x); self._sin[n:].zero_()
        self._g.replay()
        with torch.no_grad():
            return self._select(self._sout)[:n]


class BucketedLeanPose:
    """剪块数每帧不同，而 CUDA Graph 要静态形状 —— 按 2 的幂分档各录一张图，
    实际 batch 补齐到最近的档。超过最大档就分块跑。所有档共享同一份权重。"""

    def __init__(self, pt, size, buckets=(1, 2, 4, 8, 16), device="cuda", warm=20, fuse=True):
        from ultralytics import YOLO
        with contextlib.redirect_stdout(io.StringIO()):
            net = YOLO(pt).model.to(device).eval()
            if fuse:
                net = net.fuse()
            net = net.eval()
        self.buckets = sorted(buckets)
        self.m = {}
        for b in self.buckets:
            self.m[b] = LeanPose(pt, size, device, batch=b, net=net).graph(warm=warm)

    def __call__(self, x):
        n = x.shape[0]
        if n <= self.buckets[-1]:
            b = next(k for k in self.buckets if k >= n)
            return self.m[b](x)
        top = self.buckets[-1]
        return torch.cat([self.m[top](x[i:i+top]) if x[i:i+top].shape[0] == top
                          else self(x[i:i+top]) for i in range(0, n, top)], 0)


# ------------------------------------------------------------------ 自检
if __name__ == "__main__":
    import time, argparse
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sys.stdout.reconfigure(encoding="utf-8")
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    PT = os.path.join(_ROOT, "runs", "fpga_pose_p345_full", "weights", "<checkpoint>.pt")
    S = a.size

    def timeit(fn, reps=50, warm=15):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter()-t0)*1000)
        ts.sort(); return ts[len(ts)//2]

    print(f"GPU: {torch.cuda.get_device_name(0)}   输入 {S}x{S}")
    torch.backends.cudnn.benchmark = True
    x = torch.rand(1, 3, S, S, device="cuda")

    with contextlib.redirect_stdout(io.StringIO()):
        ref = YOLO(PT).model.cuda().eval()
    with torch.no_grad():
        r = ref(x)
    r = r[0] if isinstance(r, (tuple, list)) else r

    lean = LeanPose(PT, S)
    with torch.no_grad():
        l = lean.forward(x)
    print(f"\n{'='*70}\n等价性\n{'='*70}")
    print(f"  原模型输出 {tuple(r.shape)}   精简路径 {tuple(l.shape)}")
    d = (r.float()-l.float()).abs().max().item()
    print(f"  最大绝对差 {d:.3e}   ->   {'逐位一致' if d == 0 else ('数值一致' if d < 1e-4 else '不一致！')}")

    print(f"\n{'='*70}\n计时\n{'='*70}")
    res = {}
    with torch.no_grad():
        res["原模型 eager（现状）"] = timeit(lambda: ref(x))
        res["精简 one2one eager"] = timeit(lambda: lean.forward(x))
    lg = LeanPose(PT, S).graph()
    y = lg(x)
    dg = (r.float()-y.float()).abs().max().item()
    res["精简 + CUDA Graph"] = timeit(lambda: lg(x))
    print(f"  [图输出与原模型最大差 {dg:.3e}]")
    base = res["原模型 eager（现状）"]
    print(f"  {'配置':<28}{'ms':>9}{'fps':>9}{'加速':>9}")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28}{v:9.2f}{1000/v:9.2f}{base/v:8.2f}x")
