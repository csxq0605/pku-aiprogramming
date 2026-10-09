"""
Tiny ImageNet 实验的公共部分：命令行参数、数据、超参数、训练循环与计时、结果保存
TinyTensor（imagenet_train.py）与 PyTorch（torch_imagenet.py）共用，保证两边的数据顺序、增强、学习率等完全一致

计时只在 epoch / 基准测试的首尾同步 GPU，step 之间不插入任何同步：
训练 loss / 正确数在 GPU 上累加，epoch 结束时才读回
"""
import argparse
import json
import os
import time

import numpy as np

DATA_DIR = os.path.expanduser("~/data/tiny-imagenet-200")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "imagenet")
MEAN = [0.4802, 0.4481, 0.3975]
STD = [0.2764, 0.2689, 0.2816]
PAD = 4          # 随机裁剪：四周补 4 个像素再裁回 64x64
EVAL_BATCH = 250

# 每个网络的默认优化器与超参数（两边相同）
DEFAULTS = {
    "vgg16": dict(optimizer="sgd", lr=0.05, wd=5e-4),
    "resnet18": dict(optimizer="sgd", lr=0.1, wd=5e-4),
    "resnext26": dict(optimizer="sgd", lr=0.1, wd=5e-4),
    "vit_tiny": dict(optimizer="adamw", lr=1e-3, wd=0.05),
}


def parse_args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18", choices=sorted(DEFAULTS))
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--warmup-epochs", type=float, default=None, help="默认：SGD 1 个 epoch，AdamW 5 个 epoch（不超过总数的 1/5）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-graph", dest="graph", action="store_false", help="逐个 step 发射，不使用 CUDA Graph")
    p.add_argument("--chunk", type=int, default=50, help="一张 CUDA Graph 里录的 step 数")
    p.add_argument("--bench", type=int, default=0, help=">0 时只测吞吐：预热后计时这么多个 step，不训练完整 epoch")
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--train-limit", type=int, default=0, help="只用前 N 张训练图（调试用）")
    p.add_argument("--tag", default="")
    if extra:
        extra(p)
    args = p.parse_args()
    d = DEFAULTS[args.arch]
    args.optimizer = d["optimizer"]
    args.lr = d["lr"] if args.lr is None else args.lr
    args.wd = d["wd"] if args.wd is None else args.wd
    if args.warmup_epochs is None:
        args.warmup_epochs = 1.0 if args.optimizer == "sgd" else min(5.0, args.epochs / 5)
    return args


def load_data(train_limit=0):
    def load(name):
        return np.load(os.path.join(DATA_DIR, name))
    tx, ty, vx, vy = load("train_x.npy"), load("train_y.npy"), load("val_x.npy"), load("val_y.npy")
    if train_limit:
        tx, ty = tx[:train_limit], ty[:train_limit]
    return tx, ty.astype(np.int32), vx, vy.astype(np.int32)


def schedule(args, steps_per_epoch):
    """学习率：线性 warmup（第 t 步为 lr * (t+1) / warmup），之后余弦衰减到 0；两边都在 GPU 上按步数计算"""
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(round(args.warmup_epochs * steps_per_epoch)))
    return dict(base_lr=args.lr, warmup_steps=warmup, total_steps=total, final_ratio=0.0)


def epoch_order(seed, epoch, n):
    return np.random.default_rng(seed * 1000 + epoch).permutation(n).astype(np.int32)


def aug_seed(seed):
    return 12345 + seed


def init_path(arch, seed):
    return os.path.join(RESULTS, "init", "{}_seed{}.npz".format(arch, seed))


def run(trainer, args, framework):
    """
    trainer 需要提供：
      begin_epoch(order)      上传本 epoch 的打乱顺序，batch 计数清零
      train_steps(k)          连续训练 k 个 step（不同步）
      train_stats()           读回并清零本 epoch 在 GPU 上累加的 (loss 总和, 正确数)
      evaluate()              验证集 (平均 loss, 准确率)
      sync()
    """
    n_train = trainer.n_train
    steps_per_epoch = n_train // args.batch
    os.makedirs(RESULTS, exist_ok=True)
    name = "{}_{}_{}{}".format(args.arch, args.dtype, framework["key"], ("_" + args.tag) if args.tag else "")
    record = dict(arch=args.arch, dtype=args.dtype, framework=framework["name"], batch=args.batch,
                  optimizer=args.optimizer, lr=args.lr, wd=args.wd, graph=args.graph, chunk=args.chunk,
                  params=trainer.num_params)

    if args.bench:
        # 吞吐测试：先跑 warmup 个 step（包括第一个 eager step、CUDA Graph 录制或 torch.compile 编译），再计时
        trainer.begin_epoch(epoch_order(args.seed, 0, n_train))
        t0 = time.perf_counter()
        trainer.train_steps(args.bench_warmup)
        trainer.sync()
        warm = time.perf_counter() - t0
        assert args.bench_warmup + args.bench <= steps_per_epoch, "bench 步数超过一个 epoch"
        t0 = time.perf_counter()
        trainer.train_steps(args.bench)
        trainer.sync()
        elapsed = time.perf_counter() - t0
        loss_sum, correct = trainer.train_stats()
        n = (args.bench_warmup + args.bench) * args.batch
        record.update(bench_steps=args.bench, warmup_seconds=warm, seconds=elapsed,
                      ms_per_step=elapsed / args.bench * 1e3, images_per_sec=args.bench * args.batch / elapsed,
                      train_loss=loss_sum / n, peak_mem_mb=trainer.peak_memory_mb())
        print("{} | {} {} | {:.2f} ms/step | {:.0f} img/s | warmup {:.1f}s | loss {:.3f} | mem {} MB".format(
            framework["name"], args.arch, args.dtype, record["ms_per_step"], record["images_per_sec"], warm,
            record["train_loss"], record["peak_mem_mb"]))
        path = os.path.join(RESULTS, "bench_" + name + ".json")
    else:
        history = []
        total_time = 0.0
        for epoch in range(args.epochs):
            trainer.begin_epoch(epoch_order(args.seed, epoch, n_train))
            trainer.sync()
            t0 = time.perf_counter()
            trainer.train_steps(steps_per_epoch)
            trainer.sync()
            dt = time.perf_counter() - t0
            total_time += dt
            loss_sum, correct = trainer.train_stats()
            n = steps_per_epoch * args.batch
            val_loss, val_acc = trainer.evaluate()
            history.append(dict(epoch=epoch + 1, seconds=dt, train_loss=loss_sum / n, train_acc=correct / n,
                                val_loss=val_loss, val_acc=val_acc))
            print("epoch {:3d} | {:7.1f}s | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f}".format(
                epoch + 1, dt, loss_sum / n, correct / n, val_loss, val_acc), flush=True)
        record.update(epochs=args.epochs, history=history, train_seconds=total_time,
                      best_val_acc=max(h["val_acc"] for h in history), final_val_acc=history[-1]["val_acc"],
                      peak_mem_mb=trainer.peak_memory_mb())
        path = os.path.join(RESULTS, "train_" + name + ".json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print("saved", path)
    return record
