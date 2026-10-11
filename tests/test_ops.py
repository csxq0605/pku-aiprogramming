"""
算子正确性测试（contiguous_format 那一组 kernel 与框架本身）：与 numpy 参考实现对比前向和梯度，
对整个 CNN 做有限差分检查，并核对融合层、CUDA Graph、优化器与逐算子 / numpy 写法一致
用法: python test_ops.py（也可以用 pytest 运行）
"""
import _paths  # noqa: F401
import numpy as np

import tinytensor as tt
import tinytensor.nn.functional as F
from tinytensor import nn, optim
from mnist_common import SGD, Adam
from tiny_trainer import TinyTrainer

rng = np.random.default_rng(0)


def close(a, b, tol=1e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))
    assert err < tol, "max relative error {:.3e}".format(err)


def leaf(x):
    """需要梯度的输入"""
    return tt.tensor(x, requires_grad=True)


def grads_of(output, weights_R, inputs):
    """计算 sum(output * R) 对 inputs 的梯度"""
    R = tt.Tensor(weights_R.astype(np.float32), requires_grad=False)
    tt.summation(tt.multiply(output, R)).backward()
    return [x.grad.numpy() for x in inputs]


def np_conv(x, w, pad):
    n, c, h, wd = x.shape
    co, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = h + 2 * pad - kh + 1, wd + 2 * pad - kw + 1
    out = np.zeros((n, co, oh, ow))
    for i in range(kh):
        for j in range(kw):
            out += np.einsum("nchw,oc->nohw", xp[:, :, i:i + oh, j:j + ow], w[:, :, i, j])
    return out


def np_conv_grad(x, w, pad, R):
    n, c, h, wd = x.shape
    co, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = R.shape[2], R.shape[3]
    gw = np.zeros_like(w, dtype=np.float64)
    gxp = np.zeros_like(xp, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            gw[:, :, i, j] = np.einsum("nohw,nchw->oc", R, xp[:, :, i:i + oh, j:j + ow])
            gxp[:, :, i:i + oh, j:j + ow] += np.einsum("nohw,oc->nchw", R, w[:, :, i, j])
    return gxp[:, :, pad:pad + h, pad:pad + wd], gw


def test_linear():
    x = rng.standard_normal((7, 5)).astype(np.float32)
    w = rng.standard_normal((5, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    X, W, B = leaf(x), nn.Parameter(data=w), nn.Parameter(data=b)
    out = F.linear(X, W, B)
    close(out.numpy(), x @ w + b)
    R = rng.standard_normal((7, 3))
    gx, gw, gb = grads_of(out, R, [X, W, B])
    close(gx, R @ w.T)
    close(gw, x.T @ R)
    close(gb, R.sum(0))


def test_relu():
    x = rng.standard_normal((4, 6)).astype(np.float32)
    X = leaf(x)
    out = F.relu(X)
    close(out.numpy(), np.maximum(x, 0))
    R = rng.standard_normal((4, 6))
    gx, = grads_of(out, R, [X])
    close(gx, R * (x > 0))


def test_conv2d():
    # 前四组走直接卷积（Cin*kh*kw <= 256，其中 5x5 走通用 kernel），最后一组 Cin=32 走 im2col + GEMM
    for shape, wshape, pad in [((2, 1, 6, 6), (3, 1, 3, 3), 1), ((3, 3, 7, 5), (4, 3, 3, 3), 1),
                               ((2, 2, 5, 5), (2, 2, 3, 3), 0), ((2, 2, 7, 6), (3, 2, 5, 5), 2),
                               ((2, 32, 6, 5), (4, 32, 3, 3), 1)]:
        x = rng.standard_normal(shape).astype(np.float32)
        w = rng.standard_normal(wshape).astype(np.float32)
        X, W = leaf(x), nn.Parameter(data=w)
        out = F.conv2d(X, W, padding=pad)
        ref = np_conv(x, w, pad)
        close(out.numpy(), ref)
        R = rng.standard_normal(ref.shape)
        gx, gw = grads_of(out, R, [X, W])
        rgx, rgw = np_conv_grad(x, w, pad, R)
        close(gx, rgx)
        close(gw, rgw)


def test_max_pool2d():
    x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
    X = leaf(x)
    out = F.max_pool2d(X, 2)
    blocks = x.reshape(2, 3, 3, 2, 3, 2)
    ref = blocks.max(axis=(3, 5))
    close(out.numpy(), ref)
    R = rng.standard_normal(ref.shape)
    gx, = grads_of(out, R, [X])
    mask = (blocks == ref[:, :, :, None, :, None])
    close(gx, (mask * R[:, :, :, None, :, None]).reshape(x.shape))


def test_cross_entropy():
    z = (rng.standard_normal((16, 10)) * 3).astype(np.float32)
    y = rng.integers(0, 10, 16)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for loss_fn, Z in ((F.cross_entropy, leaf(z)), (F.cross_entropy_reference, tt.Tensor(z))):
        loss = loss_fn(Z, tt.tensor(y))
        close(loss.numpy(), -np.mean(np.log(p[np.arange(16), y])))
        loss.backward()
        close(Z.grad.numpy(), (p - np.eye(10)[y]) / 16)


def test_view_and_reshape_on_gpu():
    x = rng.standard_normal((4, 3, 2, 2)).astype(np.float32)
    for op in (lambda t: tt.reshape(t, (4, -1)), lambda t: t.view(4, -1), lambda t: t.flatten()):
        X = leaf(x)
        out = op(X)
        assert out.shape == (4, 12)
        close(out.numpy(), x.reshape(4, -1))
        R = rng.standard_normal((4, 12))
        gx, = grads_of(out, R, [X])
        close(gx, R.reshape(x.shape))


def test_optimizers_match_numpy():
    for reference, make in ((SGD(0.1), lambda p: optim.SGD(p, lr=0.1)), (Adam(0.01), lambda p: optim.Adam(p, lr=0.01))):
        p_np = rng.standard_normal((5, 7)).astype(np.float32)
        p_gpu = nn.Parameter(data=p_np.copy())
        opt = make([p_gpu])
        for _ in range(3):
            g = rng.standard_normal((5, 7)).astype(np.float32)
            reference.step([p_np], [g])
            p_gpu.grad = tt.tensor(g)
            opt.step()
        close(p_gpu.numpy(), p_np, tol=1e-5)


def test_kaiming_uniform_init():
    def make(seed):
        layer = nn.Linear(784, 100)
        nn.init.kaiming_uniform_(layer.weight)
        layer.manual_seed(seed)
        return layer.parameters()[0].numpy()

    fan_in = 784
    w = make(7)
    bound = np.sqrt(6.0 / fan_in)
    assert np.all(np.abs(w) <= bound + 1e-6)
    assert abs(w.mean()) < 0.01 * bound
    assert abs(w.std() - np.sqrt(2.0 / fan_in)) < 0.02 * np.sqrt(2.0 / fan_in)
    # 同一种子可复现，不同种子不同
    assert np.array_equal(w, make(7))
    assert not np.array_equal(w, make(8))
    # 不同 block 中线程号相同的位置不应重复（原实现每 256 个元素重复一次）
    flat = w.ravel()
    assert not np.allclose(flat[:256], flat[256:512])


def test_cnn_finite_difference():
    """整网梯度与中心差分对比（float32，容差放宽）"""
    trainer = TinyTrainer("cnn", seed=3)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 4)
    loss, _ = trainer.loss(x, y)
    loss.backward()

    def loss_with(w, data):
        w.raw.copy_from(tt.tensor(data).realize_cached_data(), 0)
        return trainer.loss(x, y)[0].numpy().item()

    for k, w in enumerate(trainer.weights):
        g = w.grad.numpy()
        data = w.numpy().copy()
        for idx in np.argsort(-np.abs(g).ravel())[:3]:
            idx = np.unravel_index(idx, data.shape)
            old = data[idx]
            eps = 1e-3
            data[idx] = old + eps
            lp = loss_with(w, data)
            data[idx] = old - eps
            lm = loss_with(w, data)
            data[idx] = old
            loss_with(w, data)
            numeric = (lp - lm) / (2 * eps)
            assert abs(numeric - g[idx]) <= 2e-2 * max(abs(numeric), 1e-2), \
                "param {} {}: analytic {:.5f} numeric {:.5f}".format(k, idx, g[idx], numeric)


def test_fused_layers_match_unfused():
    """融合层（conv+ReLU+池化、FC+ReLU）与逐层写法训练结果一致"""
    x = rng.standard_normal((40, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 40)
    for arch in ("mlp", "cnn"):
        results = []
        for fused in (False, True):
            trainer, opt = TinyTrainer(arch, seed=5, use_graph=False, fused=fused), SGD(0.05)
            trainer.begin_epoch(x, y, np.arange(40))
            losses = [trainer.train_batch(i, 8, opt).numpy().item() for i in range(0, 40, 8)]
            results.append((losses, [w.numpy() for w in trainer.weights]))
        close(results[1][0], results[0][0], tol=1e-5)
        for a, b in zip(results[0][1], results[1][1]):
            close(b, a, tol=1e-5)


def test_conv_relu_max_pool2d():
    for shape, wshape, pad in [((2, 1, 8, 8), (3, 1, 3, 3), 1), ((3, 3, 7, 9), (10, 3, 3, 3), 1),
                               ((2, 2, 9, 8), (3, 2, 5, 5), 2)]:
        x = rng.standard_normal(shape).astype(np.float32)
        w = rng.standard_normal(wshape).astype(np.float32)
        R = rng.standard_normal((shape[0], wshape[0], (shape[2] + 2 * pad - wshape[2] + 1) // 2,
                                 (shape[3] + 2 * pad - wshape[3] + 1) // 2))
        outs = []
        for fused in (False, True):
            X, W = leaf(x), nn.Parameter(data=w)
            out = (F.conv_relu_max_pool2d(X, W, padding=pad) if fused else
                   F.max_pool2d(F.relu(F.conv2d(X, W, padding=pad)), 2))
            outs.append([out.numpy()] + grads_of(out, R, [X, W]))
        for a, b in zip(*outs):
            close(b, a)


def test_cuda_graph_matches_eager():
    """CUDA Graph 重放与逐算子执行的训练结果一致（SGD / Adam，MLP / CNN）"""
    x = rng.standard_normal((40, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 40)
    order, shuffled = np.arange(40), rng.permutation(40)
    for arch in ("mlp", "cnn"):
        for make_opt in (lambda: SGD(0.05), lambda: Adam(1e-3)):
            results = []
            # 逐算子 / 每图 1 个 step / 每图 2 个 step（5 个 batch 时会同时用到 2 步图和 1 步图）
            for use_graph, chunk in ((False, 1), (True, 1), (True, 2)):
                trainer, opt = TinyTrainer(arch, seed=5, use_graph=use_graph, chunk=chunk), make_opt()
                for epoch in range(2):
                    trainer.begin_epoch(x, y, shuffled if epoch else order)
                    trainer.train_steps(0, 5, 8, opt)
                trainer.begin_epoch(x, y, order)
                trainer.train_steps(2, 3, 8, opt)
                trainer.train_batch(36, 4, opt)
                trainer.sync()
                results.append([w.numpy() for w in trainer.weights])
            for other in results[1:]:
                for eager, graph in zip(results[0], other):
                    close(graph, eager, tol=1e-5)


def test_module_api():
    """nn.Module：参数按声明顺序登记，state_dict 可以原样装回，no_grad 下不建计算图"""
    model = tt.models.CNN().manual_seed(1)
    names = [name for name, _ in model.named_parameters()]
    assert names == ["conv1.weight", "conv2.weight", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    state = model.state_dict()
    other = tt.models.CNN().manual_seed(2)
    assert not np.array_equal(other.state_dict()["conv1.weight"], state["conv1.weight"])
    other.load_state_dict(state)
    x = tt.tensor(rng.standard_normal((4, 1, 28, 28)).astype(np.float32))
    with tt.no_grad():
        a, b = model(x), other(x)
    assert a.is_leaf() and not a.requires_grad
    close(b.numpy(), a.numpy(), tol=1e-6)
    assert not model(x).is_leaf()


def test_training_step_keeps_graph_small():
    """参数是计算图之外的叶子：更新后不会把历史计算图串起来"""
    trainer, opt = TinyTrainer("mlp", seed=0), SGD(0.1)
    x = rng.standard_normal((8, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 8)
    for _ in range(3):
        loss = trainer.train_step(x, y, opt)
    assert all(w.is_leaf() for w in trainer.weights)
    assert len(tt.autograd.find_topo_sort([loss])) <= 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
