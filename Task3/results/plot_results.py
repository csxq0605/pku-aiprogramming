"""
根据 json/ 下的训练记录画 TinyTensor 与 PyTorch 的对比图（WSL 同一环境，bench_wsl.sh 生成）
用法: python plot_results.py  ->  生成 compare.png
"""
import json
import os

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["hatch.linewidth"] = 1.2

HERE = os.path.dirname(os.path.abspath(__file__))
TINY, TORCH = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def load(name):
    with open(os.path.join(HERE, "json", name + ".json")) as f:
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

# 测试准确率曲线：两边都用 CUDA Graph（逐算子模式的数值几乎相同）
for ax, arch in zip(axes[:2], ("mlp", "cnn")):
    finals = {key: 1 - load("{}_{}_graph".format(key, arch))[-1]["test_err"] for key in ("tinytensor", "pytorch")}
    for key, label, color, marker in (("tinytensor", "TinyTensor", TINY, "o"), ("pytorch", "PyTorch", TORCH, "s")):
        # 末端标注：数值较高的一条往上放，较低的往下放，避免重叠
        dy = 4 if finals[key] >= max(finals.values()) and key == max(finals, key=finals.get) else -10
        h = load("{}_{}_graph".format(key, arch))
        epochs = [r["epoch"] for r in h]
        acc = [(1 - r["test_err"]) * 100 for r in h]
        ax.plot(epochs, acc, color=color, linewidth=2, marker=marker, markersize=6, label=label)
        ax.annotate("{:.2f}%".format(acc[-1]), (epochs[-1], acc[-1]), textcoords="offset points",
                    xytext=(6, dy), color=INK, fontsize=8)
    style(ax, "{} 测试准确率 (%)".format(arch.upper()))
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

# 每个 epoch 的训练时间：颜色区分框架，纹理区分执行方式（实心 = CUDA Graph）
ax = axes[2]
configs = [("tinytensor", "graph", "TinyTensor + CUDA Graph", TINY, None),
           ("tinytensor", "eager", "TinyTensor 逐算子", TINY, "///"),
           ("pytorch", "graph", "PyTorch + CUDA Graph", TORCH, None),
           ("pytorch", "compile_ro", "PyTorch torch.compile (reduce-overhead)", TORCH, "xxx"),
           ("pytorch", "compile", "PyTorch torch.compile (default)", TORCH, "..."),
           ("pytorch", "eager", "PyTorch eager（默认）", TORCH, "///")]
width = 0.14
for i, (fw, mode, label, color, hatch) in enumerate(configs):
    times = [steady(load("{}_{}_{}".format(fw, arch, mode))) for arch in ("mlp", "cnn")]
    xs = [j + (i - (len(configs) - 1) / 2) * (width + 0.01) for j in range(2)]
    bars = ax.bar(xs, times, width=width, color="white" if hatch else color, edgecolor=color,
                  hatch=hatch, linewidth=1.5, label=label)
    for b, t in zip(bars, times):
        ax.annotate("{:.3f}".format(t), (b.get_x() + b.get_width() / 2, t), textcoords="offset points",
                    xytext=(0, 3), ha="center", color=INK, fontsize=7)
style(ax, "每个 epoch 训练时间 (s，第 2 个 epoch 起平均，越低越好)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["MLP", "CNN"])
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "compare.png"), dpi=150)
print("saved", os.path.join(HERE, "compare.png"))
