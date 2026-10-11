"""
TinyTensor 的 MNIST 训练器：tinytensor.models 里的 MLP / CNN + 常驻 GPU 的数据 + CUDA Graph
训练数据只上传一次，batch 按 GPU 上的步数取出；默认把连续 chunk 个 step 录成一张 CUDA Graph 重放（--no-graph 关闭）
"""
import numpy as np

import tinytensor as tt
from tinytensor import nn, optim
from tinytensor.models import CNN, MLP
from tinytensor.utils.data import TensorLoader
from mnist_common import save_init


class TinyTrainer:
    def __init__(self, arch, seed=0, export_init=False, use_graph=True, chunk=100, fused=True):
        # fused=True 时使用融合层（conv+ReLU+池化、FC+ReLU），数学上与逐层写法完全相同
        self.arch = arch
        self.model = (CNN if arch == "cnn" else MLP)(fused=fused).manual_seed(seed)
        self.weights = self.model.parameters()
        self.criterion = nn.CrossEntropyLoss()
        if export_init:
            save_init(arch, seed, [w.numpy() for w in self.weights])
        self.use_graph, self.chunk = use_graph, chunk
        self.loader = None
        self.optimizers, self.runners = {}, {}

    def prepare(self, X):
        # MLP 的输入在上传前就展平（模型里的 Flatten 对二维输入什么都不做）
        return X.reshape(X.shape[0], -1) if self.arch == "mlp" else X

    def optimizer(self, spec):
        """mnist_common 里的 SGD / Adam 只描述超参数（PyTorch 对照脚本也读它），这里换成 tinytensor.optim 的优化器"""
        if id(spec) not in self.optimizers:
            if spec.__class__.__name__ == "Adam":
                opt = optim.Adam(self.weights, lr=spec.lr, betas=(spec.beta1, spec.beta2), eps=spec.eps)
            else:
                opt = optim.SGD(self.weights, lr=spec.lr)
            self.optimizers[id(spec)] = (opt, spec)   # 留住 spec，保证 id 不被复用
        return self.optimizers[id(spec)][0]

    # ---------- 一个训练 step ----------
    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        optimizer.step()
        # 返回 GPU 上的 loss，不在每个 step 同步；需要数值时再调用 .numpy()
        return loss

    def loss(self, X, y):
        """numpy 输入的前向，返回 (loss, logits)"""
        logits = self.model(tt.tensor(self.prepare(X)))
        return self.criterion(logits, tt.tensor(y)), logits

    def train_step(self, X, y, spec):
        """numpy 输入的单步训练（不使用 CUDA Graph）"""
        return self.step(tt.tensor(self.prepare(X)), tt.tensor(y), self.optimizer(spec))

    # ---------- epoch ----------
    def begin_epoch(self, X, y, order):
        """训练集只上传一次；每个 epoch 只上传打乱顺序，并把 GPU 上的步数清零"""
        if self.loader is None:
            self.loader = TensorLoader(self.prepare(X), y, batch_size=None)
        self.loader.set_epoch(order)
        self.next_step = 0

    def train_steps(self, first, steps, batch, spec):
        """训练打乱后数据集中第 first 到 first + steps - 1 个完整 batch"""
        if first != self.next_step:
            self.loader.seek(first)
        self.next_step = first + steps
        key = (batch, id(spec))
        if key not in self.runners:
            optimizer = self.optimizer(spec)

            def one_step():
                self.loader.batch_size = batch
                return self.step(*self.loader.next(), optimizer)

            # 录制期间写出的参数梯度要一直保持存活（见 GraphRunner）
            self.runners[key] = tt.cuda.GraphRunner(one_step, self.chunk, self.use_graph,
                                                    keep=lambda: [w._g for w in self.weights])
        return self.runners[key].run(steps)

    def train_batch(self, start, count, spec):
        """训练打乱后数据集中 [start, start + count) 这一批（任意位置，逐算子执行）"""
        return self.step(*self.loader.slice(start, count), self.optimizer(spec))

    def evaluate(self, X, y, batch=1000):
        total_loss, wrong = 0.0, 0
        with tt.no_grad():
            for i in range(0, X.shape[0], batch):
                loss, logits = self.loss(X[i:i + batch], y[i:i + batch])
                n = min(batch, X.shape[0] - i)
                total_loss += loss.numpy().item() * n
                wrong += int(np.sum(logits.numpy().argmax(axis=1) != y[i:i + batch]))
        return total_loss / X.shape[0], wrong / X.shape[0]

    def sync(self):
        tt.cuda.synchronize()
