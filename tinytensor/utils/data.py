"""
常驻 GPU 的数据加载：整个数据集上传一次，每个 step 由一个 kernel 按 GPU 上的步数取出 batch（并做数据增强）。
取 batch 不需要 CPU 参与，所以连续的训练 step 可以录进同一张 CUDA Graph
"""
import numpy as np

from .. import _state
from .._C import myNN, myTensor
from ..tensor import Tensor, tf, ti


def _int(a):
    return ti(np.ascontiguousarray(a, dtype=np.int32), "gpu")


class TensorLoader:
    """
    (X, y) 以 fp32 / int32 常驻 GPU；next() 取出打乱顺序里的下一个 batch。
    取 batch 与推进步数合在一个 kernel 里（WSL 下每个 kernel 都有固定的提交开销）
    """

    def __init__(self, X, y, batch_size):
        self.X = tf(np.ascontiguousarray(X, dtype=np.float32), "gpu")
        self.y = _int(y)
        self.batch_size = batch_size
        self.order = ti([X.shape[0]], "gpu")
        self.step = ti([2], "gpu").zeros()   # [步数, kernel 内部的完成计数]
        self._buffers = {}

    def __len__(self):
        return self.X.get_shape()[0]

    def set_epoch(self, order):
        """上传这个 epoch 的打乱顺序，步数清零"""
        self.order.copy_from(_int(order), 0)
        self.step.zeros()

    def seek(self, step):
        """跳到这个 epoch 的第 step 个 batch"""
        self.step.copy_from(_int(np.array([step, 0])), 0)

    def next(self):
        b = self.batch_size
        if b not in self._buffers:
            self._buffers[b] = (tf([b] + self.X.get_shape()[1:], "gpu"), ti([b], "gpu"))
        x, y = self._buffers[b]
        myTensor.gather_xy(x, y, self.X, self.y, self.order, self.step, b)
        return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)

    def slice(self, start, count):
        """打乱顺序里 [start, start + count) 这一批（任意位置和大小，用于最后不满的一批）"""
        index = ti([count], "gpu").copy_from(self.order, start)
        x = tf([count] + self.X.get_shape()[1:], "gpu").gather_rows(self.X, index)
        y = ti([count], "gpu").gather_rows(self.y, index)
        return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)


class ImageLoader:
    """
    图像数据集以 uint8 [N, H, W, C] 常驻 GPU。next() 用一个 kernel 完成：按步数取 batch、随机裁剪（四周补 pad 个像素）、
    水平翻转、归一化、转成 fp32 / bf16，并把步数加 1；输出是 channels_last 的张量。

    augment=True : 随机裁剪 + 翻转（训练）；False：只做归一化（验证）
    mix          : None，或 (概率, mixup, cutmix)。给出时 next() 返回的标签是 (y, y2, lam)：
                   Mixup / CutMix 的两组标签与混合比例（概率为 0 时等价于不混合，用于只开标签平滑的情况）
    step         : [本 epoch 的 batch 序号, 内部计数, 全局步数, 0]，全局步数供优化器 kernel 计算学习率
    """

    def __init__(self, images, labels, batch_size, mean, std, dtype=_state.float32, augment=False, pad=4, seed=0,
                 mix=None):
        self.images, self.labels = myNN.Tensor_uint8(images, "gpu"), _int(labels)
        self.n, self.shape = images.shape[0], list(images.shape[1:])
        self.batch_size, self.mean, self.std, self.dtype = batch_size, mean, std, dtype
        self.augment, self.pad, self.seed, self.mix = augment, pad, seed, mix
        self.order = _int(np.arange(self.n))
        self.step = ti(np.zeros(4, np.int32), "gpu")
        self.x = None

    def __len__(self):
        return self.n // self.batch_size

    def set_epoch(self, order):
        """上传这个 epoch 的打乱顺序，batch 序号清零（全局步数保留；会同步一次 GPU）"""
        self.order.copy_from(_int(order), 0)
        s = self.step.numpy()
        self.step.copy_from(ti(np.array([0, 0, s[2], 0], np.int32), "gpu"), 0)

    def reset(self):
        """回到开头（验证集每次评估前调用）"""
        self.step.zeros()

    def next(self):
        b = self.batch_size
        if self.x is None:
            self.x = (myNN.Tensor_bf16 if self.dtype == _state.bfloat16 else tf)([b] + self.shape, "gpu")
            self.y = ti([b], "gpu")
            self.y2, self.lam = ti([b], "gpu"), tf([1], "gpu")
        if self.mix is not None:
            prob, mixup, cutmix = self.mix
            myNN.load_batch_mix(self.images, self.labels, self.order, self.step, self.x, self.y, self.y2, self.lam, b,
                                self.mean, self.std, self.pad, self.seed, prob, mixup, cutmix)
            target = (self.y, self.y2, self.lam)
        else:
            myNN.load_batch(self.images, self.labels, self.order, self.step, self.x, self.y, b, self.mean, self.std,
                            self.augment, self.pad, self.seed)
            target = self.y
        return Tensor(self.x, requires_grad=False, memory_format=_state.channels_last), target
