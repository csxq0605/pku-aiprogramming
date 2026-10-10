"""
TinyTensor 在 Tiny ImageNet 上训练 VGG16 / ResNet18 / ResNeXt26 / ViT-Tiny（fp32 或 bf16 混合精度）
- 整个训练集以 uint8 常驻 GPU；每个 step 由一个 kernel 按 GPU 上的步数取 batch + 随机裁剪 + 翻转 + 归一化
- 前向 / 反向走 autodiff（nn_ops），反向时中间结果用完即释放；参数梯度累加到扁平缓冲，一个 kernel 完成优化器更新
- 学习率由优化器 kernel 根据 GPU 上的步数计算，所以连续 --chunk 个 step 可以录成一张 CUDA Graph 重放
用法: python imagenet_train.py --arch resnet18 --dtype bf16 [--epochs 30 | --bench 100] [--no-graph]
"""
import os

import numpy as np

import myNN
import myTensor
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti
from autodiff import compute_gradient_of_variables
from operators import Tensor
import imagenet_common as C
from models import MODELS


class TinyTrainer:
    def __init__(self, args, data):
        self.args = args
        free0, total = myNN.mem_info()
        self.mem_base = total - free0
        tx, ty, vx, vy = data
        self.n_train = tx.shape[0]
        self.images, self.labels = myNN.Tensor_uint8(tx, "gpu"), ti(ty, "gpu")
        self.val_images, self.val_labels = myNN.Tensor_uint8(vx, "gpu"), ti(vy, "gpu")
        self.val_order = ti(np.arange(vx.shape[0], dtype=np.int32), "gpu")
        self.n_val = vx.shape[0]

        self.model = MODELS[args.arch](args.dtype, args.seed, fuse=args.fuse, num_classes=C.NUM_CLASSES)
        self.num_params = sum(int(np.prod(p.shape)) for p in self.model.params)
        os.makedirs(os.path.dirname(C.init_path(args.arch, args.seed)), exist_ok=True)
        self.model.export_init(C.init_path(args.arch, args.seed))

        B = args.batch
        self.x = (myNN.Tensor_bf16 if args.dtype == "bf16" else tf)([B, 64, 64, 3], "gpu")
        self.y = ti([B], "gpu")
        self.order = ti([self.n_train], "gpu")
        self.step = ti(np.zeros(4, np.int32), "gpu")   # [本 epoch 的 batch 序号, 内部计数, 全局步数, 0]
        self.loss_sum, self.correct = tf([1], "gpu").zeros(), ti([1], "gpu").zeros()
        self.dlogits = tf([1], "gpu")
        n = self.model.numel
        self.m = tf([n], "gpu").zeros()
        self.v = tf([n], "gpu").zeros() if args.optimizer == "adamw" else None
        self.sched = myNN.LrSchedule(**C.schedule(args, self.n_train // B))
        self.graphs, self.warm = {}, False
        self.aug_seed = C.aug_seed(args.seed)

    # ---------- 一个训练 step（全部在 GPU 上，可被 CUDA Graph 录制） ----------
    def one_step(self):
        a, mdl = self.args, self.model
        mdl.flat_g.zeros()
        myNN.load_batch(self.images, self.labels, self.order, self.step, self.x, self.y, a.batch, C.MEAN, C.STD,
                        True, C.PAD, self.aug_seed)
        logits = mdl(Tensor(self.x, requires_grad=False), training=True)
        myNN.softmax_ce(logits.realize_cached_data(), self.y, self.dlogits, 1.0 / a.batch, self.loss_sum,
                        self.correct, True)
        compute_gradient_of_variables(logits, Tensor.make_const(self.dlogits), free_graph=True)
        lowp = mdl.flat_lowp
        if a.optimizer == "sgd":
            myNN.sgd_step(mdl.flat_w, mdl.flat_g, self.m, lowp, self.step, self.sched, 0.9, a.wd, True)
        else:
            myNN.adamw_step(mdl.flat_w, mdl.flat_g, self.m, self.v, lowp, self.step, self.sched, 0.9, 0.999, 1e-8,
                            a.wd)

    def graph(self, k):
        if k not in self.graphs:
            myTensor.graph_capture_begin()
            for _ in range(k):
                self.one_step()
            self.graphs[k] = myTensor.graph_capture_end()
        return self.graphs[k]

    def train_steps(self, steps):
        if not self.args.graph:
            for _ in range(steps):
                self.one_step()
            return
        if not self.warm:
            # 第一个 step 正常执行：cuBLAS 句柄、kernel 属性等惰性初始化不能发生在录制期间
            self.warm = True
            self.one_step()
            myNN.release_pool()   # 这一步的中间结果已经释放进普通池，录制用的是私有池，先把它们还掉
            steps -= 1
        while steps > 0:
            k = self.args.chunk if steps >= self.args.chunk else steps
            self.graph(k).launch()
            steps -= k

    # ---------- epoch 管理 ----------
    def begin_epoch(self, order):
        self.order.copy_from(ti(order, "gpu"), 0)
        s = self.step.numpy()
        self.step.copy_from(ti(np.array([0, 0, s[2], 0], np.int32), "gpu"), 0)

    def train_stats(self):
        loss, correct = float(self.loss_sum.numpy()[0]), int(self.correct.numpy()[0])
        self.loss_sum.zeros()
        self.correct.zeros()
        return loss, correct

    def evaluate(self):
        B = C.EVAL_BATCH
        x = (myNN.Tensor_bf16 if self.args.dtype == "bf16" else tf)([B, 64, 64, 3], "gpu")
        y, step = ti([B], "gpu"), ti(np.zeros(4, np.int32), "gpu")
        loss, correct, dummy = tf([1], "gpu").zeros(), ti([1], "gpu").zeros(), tf([1], "gpu")
        for _ in range(self.n_val // B):
            myNN.load_batch(self.val_images, self.val_labels, self.val_order, step, x, y, B, C.MEAN, C.STD, False,
                            C.PAD, 0)
            logits = self.model(Tensor(x, requires_grad=False), training=False)
            myNN.softmax_ce(logits.realize_cached_data(), y, dummy, 1.0, loss, correct, False)
        return float(loss.numpy()[0]) / self.n_val, int(correct.numpy()[0]) / self.n_val

    def sync(self):
        myTensor.synchronize()

    def peak_memory_mb(self):
        # 显存池不归还显存，当前占用即峰值（相对于启动时的占用）
        free, total = myNN.mem_info()
        return int((total - free - self.mem_base) / 2**20)


if __name__ == "__main__":
    args = C.parse_args()
    data = C.load_data(args.train_limit)
    trainer = TinyTrainer(args, data)
    del data
    mode = ("graph" if args.graph else "eager") + ("" if args.fuse else ", unfused")
    C.run(trainer, args, dict(key="tiny_" + mode.replace(", ", "_"), name="TinyTensor ({})".format(mode)))
