"""
卷积微基准：TinyTensor（myNN）与 PyTorch（cuDNN）在 ResNet / ResNeXt 典型层上的前向、输入梯度、权重梯度耗时
用法: python bench_conv.py [--dtype bf16] [--simt]（--simt：bf16 也强制走 SIMT 版本，对照 tensor core 的收益）
"""
import os
import sys
import time

import numpy as np

if "--simt" in sys.argv:
    os.environ["TT_CONV_SIMT"] = "1"
import myNN
import myTensor
from myTensor import Tensor_float as tf
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DTYPE = "bf16" if "--dtype" in sys.argv and sys.argv[sys.argv.index("--dtype") + 1] == "bf16" else "fp32"
B = 128
LAYERS = [  # (名字, H, C, K, R, stride, groups)
    ("stem 3->64", 64, 3, 64, 3, 1, 1),
    ("r18 l1 64", 32, 64, 64, 3, 1, 1),
    ("r18 l2 down", 32, 64, 128, 3, 2, 1),
    ("r18 l2 128", 16, 128, 128, 3, 1, 1),
    ("r18 l3 256", 8, 256, 256, 3, 1, 1),
    ("r18 l4 512", 4, 512, 512, 3, 1, 1),
    ("1x1 64->256", 32, 64, 256, 1, 1, 1),
    ("1x1 256->128", 32, 256, 128, 1, 1, 1),
    ("g32 128 (4/g)", 32, 128, 128, 3, 1, 32),
    ("g32 512 (16/g)", 8, 512, 512, 3, 1, 32),
    ("vgg 64 @64", 64, 64, 64, 3, 1, 1),
]
if "--only" in sys.argv:
    ONLY = sys.argv[sys.argv.index("--only") + 1]
    LAYERS = [l for l in LAYERS if ONLY in l[0]]


def timeit(fn, sync, iters=20, reps=5):
    """预热后测 reps 轮、每轮 iters 次，取中位数（笔记本 GPU 频率波动较大）"""
    for _ in range(5):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    return float(np.median(ts))


def main():
    tdt = torch.bfloat16 if DTYPE == "bf16" else torch.float32
    print("{:16s} {:>8s} | {:>21s} | {:>21s} | {:>21s}".format("layer", "GFLOP", "fwd tiny/torch ms", "dgrad tiny/torch",
                                                               "wgrad tiny/torch"))
    tot = np.zeros(6)
    for name, H, C, K, R, s, g in LAYERS:
        pad = (R - 1) // 2
        P = (H + 2 * pad - R) // s + 1
        flop = 2.0 * B * P * P * K * R * R * C / g / 1e9
        x = np.random.randn(B, H, H, C).astype(np.float32)
        w = (np.random.randn(K, R, R, C // g) * 0.05).astype(np.float32)
        dy = np.random.randn(B, P, P, K).astype(np.float32)

        def tt(a):
            t = tf(a, "gpu")
            return myNN.to_bf16(t) if DTYPE == "bf16" else t
        X, W, DY = tt(x), tt(w), tt(dy)
        y = myNN.Tensor_bf16([1], "gpu") if DTYPE == "bf16" else tf([1], "gpu")
        dx = myNN.Tensor_bf16([1], "gpu") if DTYPE == "bf16" else tf([1], "gpu")
        dw = tf(list(w.shape), "gpu").zeros()
        sync = myTensor.synchronize
        t_f = timeit(lambda: myNN.conv_forward(X, W, y, pad, pad, s, s, g), sync)
        t_d = timeit(lambda: myNN.conv_backward_data(DY, W, dx, [B, H, H, C], pad, pad, s, s, g), sync)
        t_w = timeit(lambda: myNN.conv_backward_weight(X, DY, dw, pad, pad, s, s, g, True), sync)

        xt = torch.tensor(x, device="cuda", dtype=tdt).permute(0, 3, 1, 2)          # channels_last
        wt = torch.tensor(w, device="cuda", dtype=tdt).permute(0, 3, 1, 2)
        dyt = torch.tensor(dy, device="cuda", dtype=tdt).permute(0, 3, 1, 2)
        if DTYPE == "fp32":
            xt, wt, dyt = xt.contiguous(), wt.contiguous(), dyt.contiguous()
        csync = torch.cuda.synchronize
        conv = torch.ops.aten.convolution_backward
        p_f = timeit(lambda: F.conv2d(xt, wt, None, s, pad, 1, g), csync)
        p_d = timeit(lambda: conv(dyt, xt, wt, None, (s, s), (pad, pad), (1, 1), False, (0, 0), g,
                                  (True, False, False)), csync)
        p_w = timeit(lambda: conv(dyt, xt, wt, None, (s, s), (pad, pad), (1, 1), False, (0, 0), g,
                                  (False, True, False)), csync)
        tot += [t_f, p_f, t_d, p_d, t_w, p_w]
        print("{:16s} {:8.2f} | {:9.3f} / {:9.3f} | {:9.3f} / {:9.3f} | {:9.3f} / {:9.3f}".format(
            name, flop, t_f, p_f, t_d, p_d, t_w, p_w))
    print("{:16s} {:8s} | {:9.3f} / {:9.3f} | {:9.3f} / {:9.3f} | {:9.3f} / {:9.3f}".format("total", "", *tot))


if __name__ == "__main__":
    main()
