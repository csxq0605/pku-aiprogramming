"""
画 Tiny ImageNet（A100）的结果图：上排 4 个网络的验证准确率曲线（TinyTensor vs PyTorch，bf16），
下排 bf16 / fp32 的训练吞吐（batch 256）
数据：imagenet_a100/ 下的 train_*.json（30 epoch 完整训练）与 bench_*.json（吞吐测试）
用法: python plot_imagenet.py  ->  生成 imagenet_a100.png
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "imagenet_a100")
ARCHS = [("vgg16", "VGG16"), ("resnet18", "ResNet18"), ("resnext26", "ResNeXt26"), ("vit_tiny", "ViT-Tiny")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
# 吞吐图的 4 个系列（固定顺序、固定颜色）
SERIES = [
    ("TinyTensor (graph)", "TinyTensor + CUDA Graph", "#2a78d6"),
    ("PyTorch eager", "PyTorch eager", "#eb6834"),
    ("PyTorch graph", "PyTorch + CUDA Graph", "#1baa7b"),
    ("PyTorch compile", "PyTorch torch.compile (best mode)", "#8b5fd6"),
]


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, pattern))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def series_key(framework):
    """把 json 里的 framework 名字归到 SERIES 的某一类；compile 的各个 mode 归为一类，取最快的"""
    if framework.startswith("TinyTensor (graph)"):
        return "TinyTensor (graph)"
    if framework.startswith("PyTorch") and "compile" in framework:
        return "PyTorch compile"
    if framework.startswith("PyTorch") and "(graph)" in framework:
        return "PyTorch graph"
    if framework.startswith("PyTorch") and "(eager)" in framework:
        return "PyTorch eager"
    return None


train = load("train_*.json")
bench = load("bench_*.json")

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.45, wspace=0.28)

# ---- 上排：验证准确率曲线（bf16） ----
for col, (arch, name) in enumerate(ARCHS):
    ax = fig.add_subplot(gs[0, col])
    style(ax, "{}, bf16: val accuracy".format(name))
    for r in train:
        if r["arch"] != arch or r["dtype"] != "bf16":
            continue
        ep = [h["epoch"] for h in r["history"]]
        acc = [100 * h["val_acc"] for h in r["history"]]
        tiny = r["framework"].startswith("TinyTensor")
        ax.plot(ep, acc, color="#2a78d6" if tiny else "#eb6834", linewidth=2, linestyle="-" if tiny else "--",
                label="{} {:.1f}%".format("TinyTensor" if tiny else "PyTorch", acc[-1]))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    if col == 0:
        ax.set_ylabel("val acc (%)", color=MUTED, fontsize=9)
    if ax.lines:
        ax.legend(frameon=False, fontsize=9, loc="lower right")

# ---- 下排：吞吐（img/s，batch 256） ----
for half, dtype in enumerate(["bf16", "fp32"]):
    ax = fig.add_subplot(gs[1, 2 * half:2 * half + 2])
    style(ax, "Training throughput, {} (img/s, batch 256, higher is better)".format(dtype))
    best = {}
    for r in bench:
        k = series_key(r["framework"])
        if r["dtype"] != dtype or k is None:
            continue
        key = (r["arch"], k)
        if key not in best or r["images_per_sec"] > best[key]["images_per_sec"]:
            best[key] = r
    x = np.arange(len(ARCHS))
    width = 0.8 / len(SERIES)
    ymax = max([v["images_per_sec"] for v in best.values()] + [1])
    for i, (k, label, color) in enumerate(SERIES):
        vals = [best[(a, k)]["images_per_sec"] if (a, k) in best else 0 for a, _ in ARCHS]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.92, color=color)
        for p, v in zip(pos, vals):
            if v:
                ax.text(p, v + ymax * 0.01, "{:.0f}".format(v), ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ARCHS], color=INK, fontsize=10)
    ax.set_ylim(0, ymax * 1.12)

fig.legend(handles=[Patch(color=c, label=l) for _, l, c in SERIES], loc="lower center", ncol=len(SERIES), frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.01))
out = os.path.join(HERE, "imagenet_a100.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
