"""
优化器（用法对应 torch.optim）

fused=False：逐参数更新（SGD 用一个 kernel 更新全部参数，Adam 每个参数一个 kernel）
fused=True ：参数在一块连续显存里（ParameterArena），一个 kernel 完成学习率计算、权重衰减、动量 / Adam 更新，
             并顺便写出 bf16 权重副本；学习率由 kernel 根据 GPU 上的步数计算（见 lr_scheduler），
             所以整个训练 step 不需要 CPU 参与，可以录进 CUDA Graph
"""
from .._C import myNN, myTensor
from ..nn.parameter import ParameterArena, zero_grad
from ..tensor import tf, ti
from . import lr_scheduler  # noqa: F401


class Optimizer:
    def __init__(self, params, lr, fused):
        self.params = list(params)
        self.lr, self.fused = lr, fused
        self.arena = None
        if fused:
            if self.params[0].arena is None:
                # 模型没有调用过 flatten_parameters()：这里把已有的参数搬进 arena
                ParameterArena(self.params).sync_shadow()
            self.arena = self.params[0].arena
            assert all(p.arena is self.arena for p in self.params), "fused optimizer needs all parameters in one arena"
            assert len(self.arena.params) == len(self.params), "fused optimizer updates the whole arena"
            # 没有学习率计划时用常数学习率；步数计数器 [本 epoch 的 batch 序号, 内部计数, 全局步数, 0]
            self.schedule = myNN.LrSchedule(base_lr=lr, warmup_steps=0, total_steps=1, final_ratio=1.0)
            self.step_counter = ti([4], "gpu").zeros()

    def zero_grad(self):
        zero_grad(self.params)

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0, nesterov=False, fused=False):
        super().__init__(params, lr, fused)
        self.momentum, self.weight_decay, self.nesterov = momentum, weight_decay, nesterov
        if fused:
            self.m = tf([self.arena.numel], "gpu").zeros()
        elif momentum or weight_decay:
            raise NotImplementedError("momentum / weight decay need fused=True")

    def step(self):
        if self.fused:
            a = self.arena
            myNN.sgd_step(a.flat_w, a.flat_g, self.m, a.flat_shadow, self.step_counter, self.schedule, self.momentum,
                          self.weight_decay, self.nesterov)
        else:
            # 一个 kernel 更新全部参数：w -= lr * g
            myTensor.axpy_many([p.raw for p in self.params], [p._g for p in self.params], -self.lr)


class Adam(Optimizer):
    """步数 t 放在 GPU 上，使整个更新可以被 CUDA Graph 录制"""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr, False)
        self.betas, self.eps = betas, eps
        self.t = self.m = self.v = None

    def step(self):
        if self.m is None:
            self.m = [tf(list(p.shape), "gpu").zeros() for p in self.params]
            self.v = [tf(list(p.shape), "gpu").zeros() for p in self.params]
            self.t = ti([1], "gpu").zeros()
        self.t.add_scalar(1)
        for p, m, v in zip(self.params, self.m, self.v):
            p.raw.adam(p._g, m, v, self.lr, self.betas[0], self.betas[1], self.eps, self.t)


class AdamW(Optimizer):
    """解耦权重衰减的 Adam（只有融合版本）"""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=True):
        if not fused:
            raise NotImplementedError("AdamW needs fused=True")
        super().__init__(params, lr, True)
        self.betas, self.eps, self.weight_decay = betas, eps, weight_decay
        self.m = tf([self.arena.numel], "gpu").zeros()
        self.v = tf([self.arena.numel], "gpu").zeros()

    def step(self):
        a = self.arena
        myNN.adamw_step(a.flat_w, a.flat_g, self.m, self.v, a.flat_shadow, self.step_counter, self.schedule,
                        self.betas[0], self.betas[1], self.eps, self.weight_decay)
