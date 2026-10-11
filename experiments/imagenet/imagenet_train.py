"""
TinyTensor 在 Tiny ImageNet / ImageNet64 上训练 VGG16 / ResNet18 / ResNeXt26 / ViT-Tiny（fp32 或 bf16 混合精度）
- 整个训练集以 uint8 常驻 GPU；每个 step 由一个 kernel 按 GPU 上的步数取 batch + 随机裁剪 + 翻转 + 归一化
- 前向 / 反向走自动微分，反向时中间结果用完即释放；参数梯度累加到扁平缓冲，一个 kernel 完成优化器更新
- 学习率由优化器 kernel 根据 GPU 上的步数计算，所以连续 --chunk 个 step 可以录成一张 CUDA Graph 重放
用法: python imagenet_train.py --arch resnet18 --dtype bf16 [--epochs 30 | --bench 100] [--no-graph]
"""
import _paths  # noqa: F401
import os

import numpy as np

import tinytensor as tt
from tinytensor import nn, optim
from tinytensor.models import MODELS
from tinytensor.utils.data import ImageLoader
import imagenet_common as C


class TinyTrainer:
    def __init__(self, args, data):
        self.args = args
        free0, total = tt.cuda.mem_info()
        self.mem_base = total - free0
        tx, ty, vx, vy = data
        self.n_train, self.n_val = tx.shape[0], vx.shape[0]
        dtype = tt.bfloat16 if args.dtype == "bf16" else tt.float32
        # 只开标签平滑时也用双标签的 kernel（混合概率为 0），与 PyTorch 对照实现一致
        mix = C.mix_cfg(args) if args.regularize else None
        self.loader = ImageLoader(tx, ty, args.batch, C.MEAN, C.STD, dtype, augment=True, pad=C.PAD,
                                  seed=C.aug_seed(args.seed), mix=mix)
        self.val_loader = ImageLoader(vx, vy, C.EVAL_BATCH, C.MEAN, C.STD, dtype, pad=C.PAD)

        self.model = MODELS[args.arch](num_classes=C.NUM_CLASSES, fuse=args.fuse)
        self.model.to(dtype).manual_seed(args.seed).flatten_parameters()
        self.num_params = self.model.num_parameters()
        os.makedirs(os.path.dirname(C.init_path(args.arch, args.seed)), exist_ok=True)
        np.savez(C.init_path(args.arch, args.seed), **{name: p.numpy() for name, p in self.model.named_parameters()})

        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.regularize else 0.0)
        self.val_criterion = nn.CrossEntropyLoss()
        params = self.model.parameters()
        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True,
                                       fused=True)
        else:
            self.optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        sched = C.schedule(args, self.n_train // args.batch)
        optim.lr_scheduler.WarmupCosine(self.optimizer, sched["warmup_steps"], sched["total_steps"],
                                        sched["final_ratio"], step=self.loader.step)
        self.runner = tt.cuda.GraphRunner(self.one_step, args.chunk, args.graph)

    # ---------- 一个训练 step（全部在 GPU 上，可被 CUDA Graph 录制） ----------
    def one_step(self):
        self.optimizer.zero_grad()
        x, target = self.loader.next()
        loss = self.criterion(self.model(x), target)
        loss.backward()
        self.optimizer.step()

    def train_steps(self, steps):
        self.runner.run(steps)

    # ---------- epoch 管理 ----------
    def begin_epoch(self, order):
        self.loader.set_epoch(order)

    def train_stats(self):
        return self.criterion.statistics()

    def evaluate(self):
        self.model.eval()
        self.val_loader.reset()
        with tt.no_grad():
            for _ in range(len(self.val_loader)):
                x, y = self.val_loader.next()
                self.val_criterion(self.model(x), y)
        self.model.train()
        loss, correct = self.val_criterion.statistics()
        return loss / self.n_val, correct / self.n_val

    def sync(self):
        tt.cuda.synchronize()

    def peak_memory_mb(self):
        # 显存池不归还显存，当前占用即峰值（相对于启动时的占用）
        free, total = tt.cuda.mem_info()
        return int((total - free - self.mem_base) / 2**20)


if __name__ == "__main__":
    args = C.parse_args()
    data = C.load_data(args.train_limit)
    trainer = TinyTrainer(args, data)
    del data
    mode = ("graph" if args.graph else "eager") + ("" if args.fuse else ", unfused")
    C.run(trainer, args, dict(key="tiny_" + mode.replace(", ", "_"), name="TinyTensor ({})".format(mode)))
