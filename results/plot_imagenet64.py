"""
画 ImageNet64（1000 类，A100，bf16，batch 128）的结果图，共两张：
- imagenet64_a100_training.png：2 行 x 4 个网络。
  上排是前 10 个 epoch 的验证准确率对照（TinyTensor / PyTorch eager / torch.compile 用同一日程）；
  下排是 TinyTensor 跑完整个日程（CNN 60 epoch，ViT-Tiny 100 epoch）的验证准确率
- imagenet64_a100_time.png：每个 epoch 的稳定耗时（第 2 个 epoch 起的中位数，秒，越低越好）
数据：imagenet64_a100/ 下的 train_imagenet64_*.json
用法: python plot_imagenet64.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "imagenet64_a100")
ARCHS = [("vgg16", "VGG16"), ("resnet18", "ResNet18"), ("resnext26", "ResNeXt26"), ("vit_tiny", "ViT-Tiny")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
COMPARE_EPOCHS = 10
# 3 个系列（固定顺序、固定颜色，与 plot_tiny_imagenet.py 一致）
SERIES = [
    ("tiny", "TinyTensor + CUDA Graph", "#2a78d6", "-", "o"),
    ("eager", "PyTorch eager", "#eb6834", "--", "s"),
    ("compile", "PyTorch torch.compile", "#8b5fd6", ":", "^"),
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

# ================= 图 1：训练曲线，2 x 4 =================
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
fig.subplots_adjust(hspace=0.42, wspace=0.24)

# ---- 上排：前 10 个 epoch 的对照 ----
for col, (arch, name) in enumerate(ARCHS):
    ax = axes[0, col]
    style(ax, "{}: first {} epochs".format(name, COMPARE_EPOCHS))
    for k, label, color, ls, marker in SERIES:
        if (arch, k) not in runs:
            continue
        hist = runs[(arch, k)]["history"][:COMPARE_EPOCHS]
        ax.plot([h["epoch"] for h in hist], [100 * h["val_acc"] for h in hist], color=color, linewidth=2,
                linestyle=ls, marker=marker, markersize=4)
    ax.set_xticks(range(1, COMPARE_EPOCHS + 1))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    if col == 0:
        ax.set_ylabel("val top-1 (%)", color=MUTED, fontsize=9)

# ---- 下排：TinyTensor 的完整训练 ----
_, tiny_label, tiny_color, _, _ = SERIES[0]
for col, (arch, name) in enumerate(ARCHS):
    ax = axes[1, col]
    hist = runs[(arch, "tiny")]["history"]
    ep = [h["epoch"] for h in hist]
    acc = [100 * h["val_acc"] for h in hist]
    style(ax, "{}: TinyTensor, all {} epochs".format(name, ep[-1]))
    ax.plot(ep, acc, color=tiny_color, linewidth=2)
    ax.plot(ep[-1], acc[-1], "o", color=tiny_color, markersize=5)
    ax.annotate("{:.2f}%".format(acc[-1]), (ep[-1], acc[-1]), textcoords="offset points", xytext=(4, 8),
                ha="right", color=INK, fontsize=10)
    ax.set_xlim(0, ep[-1] * 1.03)
    ax.set_ylim(top=acc[-1] + 0.12 * (acc[-1] - acc[0]))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    if col == 0:
        ax.set_ylabel("val top-1 (%)", color=MUTED, fontsize=9)

fig.legend(handles=[Line2D([0], [0], color=c, linestyle=ls, marker=m, markersize=4, linewidth=2, label=l)
                    for _, l, c, ls, m in SERIES],
           loc="lower center", ncol=len(SERIES), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(HERE, "imagenet64_a100_training.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)

# ================= 图 2：每个 epoch 的稳定耗时 =================
fig, ax = plt.subplots(figsize=(11, 4.2))
style(ax, "Steady time per epoch (s, median of epoch 2+, lower is better)")
x = np.arange(len(ARCHS))
width = 0.8 / len(SERIES)
vals = {key: float(np.median([h["seconds"] for h in r["history"][1:]])) for key, r in runs.items()}
ymax = max(vals.values())
for i, (k, label, color, _, _) in enumerate(SERIES):
    for j, (arch, _) in enumerate(ARCHS):
        p = x[j] - 0.4 + width * (i + 0.5)
        if (arch, k) not in vals:
            ax.text(p, ymax * 0.01, "not run", ha="center", va="bottom", fontsize=9, color=MUTED)
            continue
        v = vals[(arch, k)]
        ax.bar(p, v, width * 0.92, color=color)
        ax.text(p, v + ymax * 0.01, "{:.1f}".format(v), ha="center", va="bottom", fontsize=9, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in ARCHS], color=INK, fontsize=10)
ax.set_ylim(0, ymax * 1.12)
fig.legend(handles=[Patch(color=c, label=l) for _, l, c, _, _ in SERIES], loc="lower center", ncol=len(SERIES),
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
out = os.path.join(HERE, "imagenet64_a100_time.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
