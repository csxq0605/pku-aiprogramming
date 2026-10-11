"""
画 ImageNet64（1000 类，A100，bf16，batch 128）的结果图：上排 4 个网络的验证准确率曲线
（TinyTensor 跑完整个日程；PyTorch eager / torch.compile 用同一日程跑前 10 个 epoch 作对照），
下排每个 epoch 的稳定耗时（第 2 个 epoch 起的中位数，秒，越低越好）
数据：imagenet64_a100/ 下的 train_imagenet64_*.json
用法: python plot_imagenet64.py  ->  生成 imagenet64_a100.png
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "imagenet64_a100")
ARCHS = [("vgg16", "VGG16"), ("resnet18", "ResNet18"), ("resnext26", "ResNeXt26"), ("vit_tiny", "ViT-Tiny")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
# 3 个系列（固定顺序、固定颜色，与 plot_imagenet.py 一致）
SERIES = [
    ("tiny", "TinyTensor + CUDA Graph", "#2a78d6", "-"),
    ("eager", "PyTorch eager", "#eb6834", "--"),
    ("compile", "PyTorch torch.compile", "#8b5fd6", ":"),
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


def series_key(framework):
    if framework.startswith("TinyTensor"):
        return "tiny"
    return "compile" if "compile" in framework else "eager"


runs = {}
for f in sorted(glob.glob(os.path.join(DATA, "train_imagenet64_*.json"))):
    with open(f) as fh:
        r = json.load(fh)
    runs[(r["arch"], series_key(r["framework"]))] = r

fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.45, wspace=0.28)

# ---- 上排：验证准确率曲线 ----
for col, (arch, name) in enumerate(ARCHS):
    ax = fig.add_subplot(gs[0, col])
    style(ax, "{}: val accuracy".format(name))
    for k, label, color, ls in SERIES:
        if (arch, k) not in runs:
            continue
        hist = runs[(arch, k)]["history"]
        ep = [h["epoch"] for h in hist]
        acc = [100 * h["val_acc"] for h in hist]
        ax.plot(ep, acc, color=color, linewidth=2, linestyle=ls,
                label="{} ep{} {:.1f}%".format(label.split(" +")[0], ep[-1], acc[-1]))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    if col == 0:
        ax.set_ylabel("val acc (%)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

# ---- 下排：每个 epoch 的稳定耗时 ----
ax = fig.add_subplot(gs[1, :])
style(ax, "Steady time per epoch (s, median of epoch 2+, lower is better)")
x = np.arange(len(ARCHS))
width = 0.8 / len(SERIES)
vals = {key: float(np.median([h["seconds"] for h in r["history"][1:]])) for key, r in runs.items()}
ymax = max(vals.values())
for i, (k, label, color, _) in enumerate(SERIES):
    for j, (arch, _) in enumerate(ARCHS):
        if (arch, k) not in vals:
            continue
        p, v = x[j] - 0.4 + width * (i + 0.5), vals[(arch, k)]
        ax.bar(p, v, width * 0.92, color=color)
        ax.text(p, v + ymax * 0.01, "{:.1f}".format(v), ha="center", va="bottom", fontsize=8, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in ARCHS], color=INK, fontsize=10)
ax.set_ylim(0, ymax * 1.12)

fig.legend(handles=[Patch(color=c, label=l) for _, l, c, _ in SERIES], loc="lower center", ncol=len(SERIES),
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(HERE, "imagenet64_a100.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
