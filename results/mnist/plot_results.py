"""
根据 <dir>/ 下的训练记录画 TinyTensor 与 PyTorch 的 MNIST 对比图（同一环境，experiments/mnist/bench.sh 生成）
用法: python plot_results.py [rtx4060|a100]  ->  生成 compare_<dir>.png；目录里缺少的配置自动跳过
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["hatch.linewidth"] = 1.2

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else "rtx4060"
TITLES = {"rtx4060": "RTX 4060 Laptop (WSL2)", "a100": "A100 (Colab)"}
TINY, TORCH = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def load(name):
    path = os.path.join(HERE, RUN, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["history"]


def steady(history):
    times = [r["epoch_time"] for r in history[1:]]
    return sum(times) / len(times)


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.5]})

# 测试准确率曲线：优先用 CUDA Graph 的记录（逐算子模式的数值几乎相同），没有时退回 eager
for ax, arch in zip(axes[:2], ("mlp", "cnn")):
    curves = {}
    for key in ("tinytensor", "pytorch"):
        curves[key] = load("{}_{}_graph".format(key, arch)) or load("{}_{}_eager".format(key, arch))
    finals = {key: 1 - h[-1]["test_err"] for key, h in curves.items()}
    for key, label, color, marker in (("tinytensor", "TinyTensor", TINY, "o"), ("pytorch", "PyTorch", TORCH, "s")):
        # 末端标注：数值较高的一条往上放，较低的往下放，避免重叠
        dy = 4 if key == max(finals, key=finals.get) else -10
        h = curves[key]
        epochs = [r["epoch"] for r in h]
        acc = [(1 - r["test_err"]) * 100 for r in h]
        ax.plot(epochs, acc, color=color, linewidth=2, marker=marker, markersize=6, label=label)
        ax.annotate("{:.2f}%".format(acc[-1]), (epochs[-1], acc[-1]), textcoords="offset points",
                    xytext=(6, dy), color=INK, fontsize=8)
    style(ax, "{} test accuracy (%), {}".format(arch.upper(), TITLES.get(RUN, RUN)))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

# 每个 epoch 的训练时间：颜色区分框架，纹理区分执行方式（实心 = CUDA Graph）
ax = axes[2]
configs = [("tinytensor", "graph", "TinyTensor + CUDA Graph", TINY, None),
           ("tinytensor", "eager", "TinyTensor op-by-op", TINY, "///"),
           ("pytorch", "graph", "PyTorch + CUDA Graph", TORCH, None),
           ("pytorch", "compile_ro", "PyTorch torch.compile (reduce-overhead)", TORCH, "xxx"),
           ("pytorch", "compile", "PyTorch torch.compile (default)", TORCH, "..."),
           ("pytorch", "eager", "PyTorch eager", TORCH, "///")]
configs = [c for c in configs if any(load("{}_{}_{}".format(c[0], arch, c[1])) for arch in ("mlp", "cnn"))]
width = 0.14
ymax = 0.0
for i, (fw, mode, label, color, hatch) in enumerate(configs):
    first = True
    for j, arch in enumerate(("mlp", "cnn")):
        h = load("{}_{}_{}".format(fw, arch, mode))
        if h is None:
            continue
        t = steady(h)
        ymax = max(ymax, t)
        x = j + (i - (len(configs) - 1) / 2) * (width + 0.01)
        ax.bar(x, t, width=width, color="white" if hatch else color, edgecolor=color, hatch=hatch, linewidth=1.5,
               label=label if first else None)
        first = False
        ax.annotate("{:.3f}".format(t), (x, t), textcoords="offset points", xytext=(0, 3), ha="center", color=INK,
                    fontsize=7)
style(ax, "Training time per epoch (s, mean of epoch 2+, lower is better)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["MLP", "CNN"])
ax.set_ylim(0, ymax * 1.25)
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")

fig.tight_layout()
out = os.path.join(HERE, "compare_{}.png".format(RUN))
fig.savefig(out, dpi=150)
print("saved", out)
