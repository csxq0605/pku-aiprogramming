"""Softmaxloss 的前向与反向一致性测试（纯 CPU，不需要编译 CUDA 扩展）。

运行：python3 Task3/Tinytensor/tests/test_softmaxloss.py
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "optimizer"))

# operators.py 在导入时需要编译好的 myTensor / myLayer；Softmaxloss 只用 numpy，这里用空模块占位。
for name in ("myTensor", "myLayer"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.Tensor_float = stub.Tensor_int = object
        sys.modules[name] = stub

from operators import Softmaxloss, Tensor  # noqa: E402


def loss_of(z, y):
    return Softmaxloss(Tensor(y))(Tensor(z)).realize_cached_data()[0]


def analytic_grad(z, y):
    op = Softmaxloss(Tensor(y))
    out = op(Tensor(z))
    out.realize_cached_data()
    return np.asarray(op.gradient(None, out).realize_cached_data())


def numeric_grad(z, y, eps=1e-6):
    g = np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        g[idx] = (loss_of(zp, y) - loss_of(zm, y)) / (2 * eps)
    return g


def test_value_matches_cross_entropy():
    # logits [0, 0]、标签 0：p = [0.5, 0.5]，交叉熵 = -log 0.5 = log 2
    z = np.array([[0.0, 0.0]])
    y = np.array([0])
    assert abs(loss_of(z, y) - np.log(2)) < 1e-12


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 10)) * 3
    y = rng.integers(0, 10, size=4)
    assert np.allclose(analytic_grad(z, y), numeric_grad(z, y), atol=1e-6)


def test_large_logits_are_stable():
    z = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    y = np.array([0, 0])
    loss = loss_of(z, y)
    # 第一行几乎为 0，第二行约 1000，均值约 500
    assert np.isfinite(loss) and abs(loss - 500.0) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
