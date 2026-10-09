"""
基于 TinyTensor 算子的 MLP / CNN 模型
训练数据常驻 GPU，batch 按 GPU 上的步数取出；默认把连续 chunk 个 step 录成一张 CUDA Graph 重放（--no-graph 关闭）
"""
import numpy as np

import myTensor
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti
from operators import (Tensor, conv, conv_relu_maxpool, fc, fc_relu, maxpool, relu, reshape,
                       softmaxcrossentropyloss)
from mnist_common import param_specs, save_init


def kaiming_uniform(shape, fan_in, seed):
    """在 GPU 上用 TinyTensor 自己的随机数生成 U(-b, b)，b = sqrt(6 / fan_in)，方差为 2 / fan_in"""
    bound = float(np.sqrt(6.0 / fan_in))
    return tf(list(shape), "gpu").random(-bound, bound, seed)


def to_gpu_float(X):
    return tf(np.ascontiguousarray(X, dtype=np.float32), "gpu")


def to_gpu_int(y):
    return ti(np.ascontiguousarray(y, dtype=np.int32), "gpu")


class StepRunner:
    """
    执行连续的训练 step。每个 step = 按 GPU 上的步数取 batch + 前向 + 反向 + 参数更新 + 步数加 1。
    batch 由 GPU 上的步数和打乱顺序决定（gather_batch），不需要 CPU 参与，因此使用 CUDA Graph 时
    可以把连续 chunk 个 step 录进同一张图：CPU 每 chunk 个 step 只发射一次，GPU 不再等 CPU，
    GPU 占用率也不受 CPU 降频的影响。
    录制期间创建的张量保存在 keep 中，保证图使用的显存在图的生命周期内不被释放或复用。
    """

    def __init__(self, model, optimizer, batch, use_graph, chunk):
        self.model, self.optimizer, self.batch = model, optimizer, batch
        self.use_graph, self.chunk = use_graph, max(1, chunk)
        self.x = tf([batch] + model.X_all.get_shape()[1:], "gpu")
        self.y = ti([batch], "gpu")
        self.graphs = {}
        self.warm = False

    def one_step(self):
        m = self.model
        # 取 batch 与推进步数合在一个 kernel 里（WSL 下每个 kernel 都有固定的提交开销）
        myTensor.gather_xy(self.x, self.y, m.X_all, m.y_all, m.order, m.step, self.batch)
        return m.step_on_gpu(self.x, self.y, self.optimizer)

    def graph(self, k):
        if k not in self.graphs:
            # 录制时 GPU 不执行，录完后由调用方发射
            myTensor.graph_capture_begin()
            for _ in range(k):
                loss = self.one_step()
            graph = myTensor.graph_capture_end()
            self.graphs[k] = (graph, (loss, [w.grad for w in self.model.weights]))
        return self.graphs[k]

    def run(self, steps):
        loss = None
        if not self.use_graph or not self.warm:
            # 第一个 step 正常执行：创建优化器状态、cuBLAS 句柄等惰性资源（这是一次真实的训练 step）
            self.warm = True
            for _ in range(steps if not self.use_graph else 1):
                loss = self.one_step()
            steps = 0 if not self.use_graph else steps - 1
        while steps > 0:
            k = self.chunk if steps >= self.chunk else 1
            graph, keep = self.graph(k)
            graph.launch()
            loss, steps = keep[0], steps - k
        return loss


class TinyModel:
    arch = None

    def __init__(self, seed=0, export_init=False, use_graph=True, chunk=100, fused=True):
        # 权重常驻 GPU，是计算图的叶子节点；优化器原地修改 cached_data，不会产生新的图节点
        self.weights = []
        for i, (shape, fan_in) in enumerate(param_specs(self.arch)):
            if fan_in is None:
                data = tf(list(shape), "gpu").zeros()
            else:
                data = kaiming_uniform(shape, fan_in, seed * 1000 + i)
            self.weights.append(Tensor(data))
        if export_init:
            save_init(self.arch, seed, [w.numpy() for w in self.weights])
        self.use_graph, self.chunk = use_graph, chunk
        # fused=True 时使用融合算子（conv+ReLU+池化、FC+ReLU），数学上与逐个算子完全相同
        self.fused = fused
        self.runners = {}
        self.X_all = self.y_all = None

    def logits(self, X):
        raise NotImplementedError()

    def prepare(self, X):
        return X

    def loss_on_gpu(self, X, y):
        X = Tensor(X, requires_grad=False)
        y = Tensor(y, requires_grad=False)
        logits = self.logits(X)
        return softmaxcrossentropyloss(logits, y), logits

    def loss(self, X, y):
        return self.loss_on_gpu(to_gpu_float(self.prepare(X)), to_gpu_int(y))

    def step_on_gpu(self, X, y, optimizer):
        loss, _ = self.loss_on_gpu(X, y)
        loss.backward()
        optimizer.step([w.cached_data for w in self.weights],
                       [w.grad.realize_cached_data() for w in self.weights])
        # 返回 GPU 上的 loss，不在每个 step 同步；需要数值时再调用 .numpy()
        return loss

    def train_step(self, X, y, optimizer):
        """numpy 输入的单步训练（不使用 CUDA Graph）"""
        return self.step_on_gpu(to_gpu_float(self.prepare(X)), to_gpu_int(y), optimizer)

    def begin_epoch(self, X, y, order):
        """训练集只上传一次；每个 epoch 只上传打乱顺序，并把 GPU 上的步数清零"""
        if self.X_all is None:
            self.X_all = to_gpu_float(self.prepare(X))
            self.y_all = to_gpu_int(y)
            self.order = ti([X.shape[0]], "gpu")
            self.step = ti([2], "gpu")  # [步数, gather_xy 内部完成计数]
        self.order.copy_from(to_gpu_int(order), 0)
        self.step.zeros()
        self.next_step = 0

    def train_steps(self, first, steps, batch, optimizer):
        """训练打乱后数据集中第 first 到 first + steps - 1 个完整 batch"""
        if first != self.next_step:
            self.step.copy_from(to_gpu_int(np.array([first, 0])), 0)
        self.next_step = first + steps
        key = (batch, id(optimizer))
        if key not in self.runners:
            self.runners[key] = StepRunner(self, optimizer, batch, self.use_graph, self.chunk)
        return self.runners[key].run(steps)

    def train_batch(self, start, count, optimizer):
        """训练打乱后数据集中 [start, start + count) 这一批（任意位置，逐算子执行）"""
        index = ti([count], "gpu").copy_from(self.order, start)
        X = tf([count] + self.X_all.get_shape()[1:], "gpu").gather_rows(self.X_all, index)
        y = ti([count], "gpu").gather_rows(self.y_all, index)
        return self.step_on_gpu(X, y, optimizer)

    def evaluate(self, X, y, batch=1000):
        total_loss, wrong = 0.0, 0
        for i in range(0, X.shape[0], batch):
            loss, logits = self.loss(X[i:i + batch], y[i:i + batch])
            n = min(batch, X.shape[0] - i)
            total_loss += loss.numpy().item() * n
            wrong += int(np.sum(logits.numpy().argmax(axis=1) != y[i:i + batch]))
        return total_loss / X.shape[0], wrong / X.shape[0]

    def sync(self):
        myTensor.synchronize()


class MLP(TinyModel):
    arch = "mlp"

    def prepare(self, X):
        return X.reshape(X.shape[0], -1)

    def logits(self, X):
        w1, b1, w2, b2 = self.weights
        X = fc_relu(X, w1, b1) if self.fused else relu(fc(X, w1, b1))
        return fc(X, w2, b2)


class CNN(TinyModel):
    arch = "cnn"

    def logits(self, X):
        conv1, conv2, w1, b1, w2, b2 = self.weights
        if self.fused:
            # 与下面的逐算子写法等价；FC 直接接受 [N, C, H, W] 输入并按 [N, C*H*W] 处理，省掉 reshape 的拷贝
            X = conv_relu_maxpool(X, conv1, pad_h=1, pad_w=1)
            X = conv_relu_maxpool(X, conv2, pad_h=1, pad_w=1)
            X = fc_relu(X, w1, b1)
            return fc(X, w2, b2)
        X = maxpool(relu(conv(X, weight=conv1, pad_h=1, pad_w=1)))
        X = maxpool(relu(conv(X, weight=conv2, pad_h=1, pad_w=1)))
        X = reshape(X, (X.shape[0], -1))
        X = relu(fc(X, weight=w1, bias=b1))
        return fc(X, weight=w2, bias=b2)
