"""
PyTorch 对照实验：与 TinyTensor 使用相同的网络结构、初始权重、数据顺序和优化器
初始权重读取 TinyTensor 训练时导出的 results/init/<arch>_seed<seed>.npz，因此要先运行 TinyTensor
训练集常驻 GPU，batch 按 GPU 上的步数取出（与 TinyTensor 相同）；--graph 时用 torch.cuda.CUDAGraph 把连续 --chunk 个
训练 step 录成一张图
--compile MODE 时用 torch.compile 编译训练 step（前向 + 反向 + 优化器），MODE 为 default / reduce-overhead（Inductor 融合 +
CUDA Graph）/ max-autotune / max-autotune-no-cudagraphs；编译耗时计入第 1 个 epoch
用法: python torch_baseline.py --arch cnn [--graph | --compile reduce-overhead] [--optimizer sgd --lr 0.1 ...]
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mnist_common import load_init, parse_args, run


class TorchModel:
    def __init__(self, arch, seed, device, use_graph=False, compile_mode=None, chunk=100):
        self.arch = arch
        self.device = device
        self.use_graph, self.chunk = use_graph, max(1, chunk)
        if arch == "mlp":
            # MLP 一张图录 99 个以上 step 时 graph.replay() 会报 CUDA launch failure（A100 / PyTorch 2.11 实测，
            # 50 及以下正常，且 chunk 取 1 / 10 / 20 / 50 的速度没有差别），所以限制在 50
            self.chunk = min(self.chunk, 50)
        self.compile_mode = compile_mode
        self.compiled_step = None
        self.X_all = None
        self.graphs = {}
        self.pool = None
        self.warm = False
        params = load_init(arch, seed)
        if arch == "mlp":
            w1, b1, w2, b2 = params
            # TinyTensor 的 FC 权重是 [in, out]，torch.nn.functional.linear 需要 [out, in]
            params = [w1.T, b1, w2.T, b2]
        else:
            c1, c2, w1, b1, w2, b2 = params
            params = [c1, c2, w1.T, b1, w2.T, b2]
        self.params = [torch.tensor(np.ascontiguousarray(p), device=device, requires_grad=True) for p in params]
        self.torch_optimizer = None

    def logits(self, X):
        if self.arch == "mlp":
            w1, b1, w2, b2 = self.params
            X = X.reshape(X.shape[0], -1)
            return F.linear(F.relu(F.linear(X, w1, b1)), w2, b2)
        c1, c2, w1, b1, w2, b2 = self.params
        X = F.max_pool2d(F.relu(F.conv2d(X, c1, padding=1)), 2)
        X = F.max_pool2d(F.relu(F.conv2d(X, c2, padding=1)), 2)
        X = X.reshape(X.shape[0], -1)
        return F.linear(F.relu(F.linear(X, w1, b1)), w2, b2)

    def _make_optimizer(self, optimizer):
        if optimizer.__class__.__name__ == "Adam":
            # capturable=True 时 Adam 的步数保存在 GPU 上，才能被 CUDA Graph 录制
            return torch.optim.Adam(self.params, lr=optimizer.lr, betas=(optimizer.beta1, optimizer.beta2),
                                    eps=optimizer.eps, capturable=self.use_graph or self.compile_mode is not None)
        return torch.optim.SGD(self.params, lr=optimizer.lr)

    def _step(self, X, y):
        loss = F.cross_entropy(self.logits(X), y)
        self.torch_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch_optimizer.step()
        return loss

    def begin_epoch(self, X, y, order):
        """与 TinyTensor 相同：训练集只上传一次，每个 epoch 上传打乱顺序，GPU 上的步数清零"""
        if self.X_all is None:
            self.X_all = torch.from_numpy(np.ascontiguousarray(X)).to(self.device)
            self.y_all = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
            self.order = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)
            self.step = torch.zeros(1, dtype=torch.long, device=self.device)
        self.order.copy_(torch.from_numpy(np.ascontiguousarray(order)))
        self.step.zero_()
        self.next_step = 0

    def _batch(self, index):
        return self.X_all.index_select(0, index), self.y_all.index_select(0, index)

    def _graph_step(self, batch):
        """按 GPU 上的步数取 batch（可被 CUDA Graph 录制）并训练一步"""
        index = self.order.index_select(0, self.step * batch + self.arange[:batch])
        loss = F.cross_entropy(self.logits(self.X_all.index_select(0, index)), self.y_all.index_select(0, index))
        loss.backward()
        self.torch_optimizer.step()
        self.step.add_(1)
        return loss

    def _graph(self, batch, k):
        key = (batch, k)
        if key not in self.graphs:
            graph = torch.cuda.CUDAGraph()
            self.torch_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph, pool=self.pool):
                for _ in range(k):
                    # 梯度在图里每步重新写入（不累加），与 zero_grad 后的 eager 行为一致
                    for p in self.params:
                        p.grad = None
                    loss = self._graph_step(batch)
            self.pool = graph.pool()
            self.graphs[key] = (graph, loss)
        return self.graphs[key]

    def train_steps(self, first, steps, batch, optimizer):
        """训练第 first 到 first + steps - 1 个完整 batch"""
        if self.torch_optimizer is None:
            self.torch_optimizer = self._make_optimizer(optimizer)
        if first != self.next_step:
            self.step.fill_(first)
        self.next_step = first + steps
        loss = None
        if self.use_graph:
            if not self.warm:
                # 第一个 step 在旁路 stream 上正常执行（初始化 cuDNN / cuBLAS / 优化器状态），之后才能录制
                self.arange = torch.arange(batch, device=self.device)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self.torch_optimizer.zero_grad(set_to_none=True)
                    loss = self._graph_step(batch)
                torch.cuda.current_stream().wait_stream(side)
                self.warm, steps = True, steps - 1
            while steps > 0:
                k = self.chunk if steps >= self.chunk else 1
                graph, loss = self._graph(batch, k)
                graph.replay()
                steps -= k
            return loss
        step_fn = self._step
        if self.compile_mode is not None:
            if self.compiled_step is None:
                self.compiled_step = torch.compile(self._step, mode=self.compile_mode, dynamic=False)
            step_fn = self.compiled_step
        cudagraph_trees = self.compile_mode in ("reduce-overhead", "max-autotune")
        for i in range(first, first + steps):
            if cudagraph_trees:
                # CUDA Graph 树：告诉 Inductor 新的一个 step 开始了，上一个 step 的输出缓冲可以复用
                torch.compiler.cudagraph_mark_step_begin()
            loss = step_fn(*self._batch(self.order[i * batch:(i + 1) * batch]))
        self.step.fill_(first + steps)
        return loss

    def train_batch(self, start, count, optimizer):
        """训练打乱后数据集中 [start, start + count) 这一批（逐算子执行）"""
        if self.torch_optimizer is None:
            self.torch_optimizer = self._make_optimizer(optimizer)
        return self._step(*self._batch(self.order[start:start + count]))

    def train_step(self, X, y, optimizer):
        if self.torch_optimizer is None:
            self.torch_optimizer = self._make_optimizer(optimizer)
        X = torch.from_numpy(np.ascontiguousarray(X)).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.ascontiguousarray(y)).to(self.device, non_blocking=True)
        loss = F.cross_entropy(self.logits(X), y)
        self.torch_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch_optimizer.step()
        return loss

    @torch.no_grad()
    def evaluate(self, X, y, batch=1000):
        total_loss, wrong = 0.0, 0
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(self.device)
            yb = torch.from_numpy(y[i:i + batch]).to(self.device)
            logits = self.logits(xb)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            wrong += (logits.argmax(dim=1) != yb).sum().item()
        return total_loss / X.shape[0], wrong / X.shape[0]

    def sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    arch = "cnn"
    if "--arch" in sys.argv:
        i = sys.argv.index("--arch")
        arch = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    compile_mode = None
    if "--compile" in sys.argv:
        i = sys.argv.index("--compile")
        compile_mode = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = parse_args(arch, default_optimizer="sgd", default_lr=0.1)
    args.graph = bool(args.graph)
    args.compile = compile_mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    framework = "PyTorch {} ({})".format(torch.__version__, device)
    if compile_mode:
        framework += " torch.compile mode=" + compile_mode
    run(arch, TorchModel(arch, args.seed, device, use_graph=args.graph, compile_mode=compile_mode,
                    chunk=args.chunk), args,
        framework=framework)
