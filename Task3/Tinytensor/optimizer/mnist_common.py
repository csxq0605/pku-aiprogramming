"""
MNIST 训练的公共部分：数据读取、参数初始化、优化器和训练循环
本文件只依赖 numpy，PyTorch 对照脚本也复用这里的数据和初始化
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST", "raw")


def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _find(data_dir, name):
    for candidate in (name, name + ".gz"):
        path = os.path.join(data_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def _download_with_torchvision(data_dir):
    from torchvision import datasets
    root = os.path.dirname(os.path.dirname(data_dir))
    datasets.MNIST(root=root, train=True, download=True)
    datasets.MNIST(root=root, train=False, download=True)


def parse_mnist(data_dir=DEFAULT_DATA_DIR):
    """
    读取 MNIST，按 (x / 255 - mean) / std 归一化
    返回 (X_tr, y_tr), (X_te, y_te)，X 形状为 [N, 1, 28, 28]，float32
    """
    names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    paths = [_find(data_dir, n) for n in names]
    if any(p is None for p in paths):
        _download_with_torchvision(data_dir)
        paths = [_find(data_dir, n) for n in names]
    X_tr, y_tr, X_te, y_te = [_read_idx(p) for p in paths]

    def normalize(x):
        x = x.astype(np.float32)[:, None, :, :] / 255.0
        return ((x - MNIST_MEAN) / MNIST_STD).astype(np.float32)

    return (normalize(X_tr), y_tr.astype(np.int64)), (normalize(X_te), y_te.astype(np.int64))


def param_specs(arch):
    """
    每个参数的 (形状, fan_in)；fan_in 为 None 表示 bias
    FC 权重形状为 [in, out]，卷积权重为 [out, in, kh, kw]
    """
    if arch == "mlp":
        return [((784, 100), 784), ((100,), None), ((100, 10), 100), ((10,), None)]
    if arch == "cnn":
        return [
            ((16, 1, 3, 3), 1 * 9),
            ((32, 16, 3, 3), 16 * 9),
            ((32 * 7 * 7, 128), 32 * 7 * 7), ((128,), None),
            ((128, 10), 128), ((10,), None),
        ]
    raise ValueError(arch)


INIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "init")


def init_path(arch, seed):
    return os.path.join(INIT_DIR, "{}_seed{}.npz".format(arch, seed))


def save_init(arch, seed, arrays):
    """保存 TinyTensor 生成的初始权重，供 PyTorch 对照实验使用同一起点"""
    os.makedirs(INIT_DIR, exist_ok=True)
    np.savez(init_path(arch, seed), *arrays)


def load_init(arch, seed):
    path = init_path(arch, seed)
    if not os.path.exists(path):
        raise FileNotFoundError("{} 不存在：请先运行 TinyTensor 的 mnist_{}.py --seed {} 生成初始权重".format(path, arch, seed))
    data = np.load(path)
    return [data["arr_{}".format(i)] for i in range(len(data.files))]


class SGD:
    """参数可以是 numpy 数组，也可以是 GPU 上的 Tensor_float（使用 axpy 原地更新）"""

    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        if params and all(hasattr(p, "axpy") for p in params):
            # GPU 参数：一次 kernel 更新全部参数
            import myTensor
            myTensor.axpy_many(params, grads, -self.lr)
            return
        for p, g in zip(params, grads):
            p -= self.lr * g


def _zeros_like(p):
    if hasattr(p, "get_shape"):
        return type(p)(p.get_shape(), "gpu").zeros()
    return np.zeros_like(p)


class Adam:
    """参数为 GPU 张量时，步数 t 也放在 GPU 上，使整个更新可以被 CUDA Graph 录制"""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.t_gpu = None
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [_zeros_like(p) for p in params]
            self.v = [_zeros_like(p) for p in params]
        self.t += 1
        if hasattr(params[0], "adam"):
            if self.t_gpu is None:
                from myTensor import Tensor_int
                self.t_gpu = Tensor_int([1], "gpu").zeros()
            self.t_gpu.add_scalar(1)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if hasattr(p, "adam"):
                p.adam(g, m, v, self.lr, self.beta1, self.beta2, self.eps, self.t_gpu)
                continue
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.dtype)


def make_optimizer(name, lr):
    return Adam(lr) if name == "adam" else SGD(lr)


def batch_orders(num_examples, epochs, seed):
    """每个 epoch 的打乱顺序，PyTorch 对照脚本使用同一顺序"""
    rng = np.random.default_rng(seed + 1)
    return [rng.permutation(num_examples) for _ in range(epochs)]


def train_loop(model, data, epochs, batch, optimizer, seed, log=print):
    """
    model 需要提供:
      begin_epoch(X, y, order) -> 设置这个 epoch 的打乱顺序（训练数据常驻 GPU）
      train_steps(first, steps, batch, optimizer) -> 训练第 first 到 first + steps - 1 个完整 batch
      train_batch(start, count, optimizer) -> 训练打乱后的第 [start, start + count) 个样本（最后不满的一批）
      evaluate(X, y, batch) -> (平均 loss, 错误率)
      sync() -> 等待 GPU 计算完成（用于计时）
    """
    (X_tr, y_tr), (X_te, y_te) = data
    orders = batch_orders(X_tr.shape[0], epochs, seed)
    # 预热：初始化 CUDA 上下文、cuBLAS 等，避免计入第一个 epoch 的训练时间（只做前向，不改权重）
    model.evaluate(X_te[:batch], y_te[:batch])
    history = []
    log("| Epoch | Train Loss | Train Err | Test Loss | Test Err | Epoch Time(s) |")
    for epoch in range(epochs):
        order = orders[epoch]
        model.sync()
        start = time.perf_counter()
        model.begin_epoch(X_tr, y_tr, order)
        full = X_tr.shape[0] // batch
        model.train_steps(0, full, batch, optimizer)
        if full * batch < X_tr.shape[0]:
            model.train_batch(full * batch, X_tr.shape[0] - full * batch, optimizer)
        model.sync()
        epoch_time = time.perf_counter() - start
        train_loss, train_err = model.evaluate(X_tr, y_tr)
        test_loss, test_err = model.evaluate(X_te, y_te)
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, train_err=train_err,
                            test_loss=test_loss, test_err=test_err, epoch_time=epoch_time))
        log("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |  {:>12.3f} |".format(
            epoch + 1, train_loss, train_err, test_loss, test_err, epoch_time))
    return history


def parse_args(arch, default_optimizer, default_lr):
    parser = argparse.ArgumentParser(description="MNIST {} training".format(arch.upper()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=default_optimizer)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="MNIST raw 文件所在目录")
    parser.add_argument("--save", default=None, help="把每个 epoch 的结果保存为 json")
    parser.add_argument("--no-graph", dest="graph", action="store_false",
                        help="不使用 CUDA Graph，逐个算子执行（TinyTensor 默认使用 CUDA Graph）")
    parser.add_argument("--graph", dest="graph", action="store_true",
                        help="使用 CUDA Graph（PyTorch 对照默认不使用）")
    parser.add_argument("--chunk", type=int, default=100,
                        help="CUDA Graph 模式下每张图包含的连续 step 数（1 = 每个 step 发射一次图）")
    parser.set_defaults(graph=None)
    return parser.parse_args()


def run(arch, model, args, framework):
    data = parse_mnist(args.data)
    optimizer = make_optimizer(args.optimizer, args.lr)
    print("{} {} | optimizer={} lr={} batch={} epochs={} seed={} cuda_graph={} chunk={}".format(
        framework, arch.upper(), args.optimizer, args.lr, args.batch, args.epochs, args.seed, args.graph,
        args.chunk))
    history = train_loop(model, data, args.epochs, args.batch, optimizer, args.seed)
    final = history[-1]
    total_time = sum(h["epoch_time"] for h in history)
    steady = [h["epoch_time"] for h in history[1:]] or [history[0]["epoch_time"]]
    print("Test accuracy: {:.2f}% | total train time: {:.3f}s | steady {:.3f}s/epoch (epoch 2+)".format(
        (1 - final["test_err"]) * 100, total_time, sum(steady) / len(steady)))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(dict(framework=framework, arch=arch, args=vars(args), history=history), f, indent=2)
    return history
