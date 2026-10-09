"""
算子正确性测试：与 numpy 参考实现对比前向和梯度，并对整个 CNN 做有限差分检查
用法: python test_ops.py（也可以用 pytest 运行）
"""
import numpy as np

from operators import Tensor, _gpu, conv, fc, maxpool, relu, reshape, softmaxcrossentropyloss, softmaxloss, summation, multiply
from tiny_nn import CNN, MLP

rng = np.random.default_rng(0)


def close(a, b, tol=1e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))
    assert err < tol, "max relative error {:.3e}".format(err)


def grads_of(output, weights_R, inputs):
    """计算 sum(output * R) 对 inputs 的梯度"""
    R = Tensor(weights_R.astype(np.float32), requires_grad=False)
    summation(multiply(output, R)).backward()
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


def test_fc():
    x = rng.standard_normal((7, 5)).astype(np.float32)
    w = rng.standard_normal((5, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    X, W, B = Tensor(x), Tensor(w), Tensor(b)
    out = fc(X, W, B)
    close(out.realize_cached_data(), x @ w + b)
    R = rng.standard_normal((7, 3))
    gx, gw, gb = grads_of(out, R, [X, W, B])
    close(gx, R @ w.T)
    close(gw, x.T @ R)
    close(gb, R.sum(0))


def test_relu():
    x = rng.standard_normal((4, 6)).astype(np.float32)
    X = Tensor(x)
    out = relu(X)
    close(out.realize_cached_data(), np.maximum(x, 0))
    R = rng.standard_normal((4, 6))
    gx, = grads_of(out, R, [X])
    close(gx, R * (x > 0))


def test_conv():
    # 前四组走直接卷积（Cin*kh*kw <= 256，其中 5x5 走通用 kernel），最后一组 Cin=32 走 im2col + GEMM
    for shape, wshape, pad in [((2, 1, 6, 6), (3, 1, 3, 3), 1), ((3, 3, 7, 5), (4, 3, 3, 3), 1),
                               ((2, 2, 5, 5), (2, 2, 3, 3), 0), ((2, 2, 7, 6), (3, 2, 5, 5), 2),
                               ((2, 32, 6, 5), (4, 32, 3, 3), 1)]:
        x = rng.standard_normal(shape).astype(np.float32)
        w = rng.standard_normal(wshape).astype(np.float32)
        X, W = Tensor(x), Tensor(w)
        out = conv(X, weight=W, pad_h=pad, pad_w=pad)
        ref = np_conv(x, w, pad)
        close(out.realize_cached_data(), ref)
        R = rng.standard_normal(ref.shape)
        gx, gw = grads_of(out, R, [X, W])
        rgx, rgw = np_conv_grad(x, w, pad, R)
        close(gx, rgx)
        close(gw, rgw)


def test_maxpool():
    x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
    X = Tensor(x)
    out = maxpool(X)
    blocks = x.reshape(2, 3, 3, 2, 3, 2)
    ref = blocks.max(axis=(3, 5))
    close(out.realize_cached_data(), ref)
    R = rng.standard_normal(ref.shape)
    gx, = grads_of(out, R, [X])
    mask = (blocks == ref[:, :, :, None, :, None])
    close(gx, (mask * R[:, :, :, None, :, None]).reshape(x.shape))


def test_softmax_cross_entropy():
    z = (rng.standard_normal((16, 10)) * 3).astype(np.float32)
    y = rng.integers(0, 10, 16)
    for loss_fn in (softmaxcrossentropyloss, softmaxloss):
        Z = Tensor(z)
        loss = loss_fn(Z, Tensor(y, requires_grad=False))
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        close(loss.realize_cached_data(), -np.mean(np.log(p[np.arange(16), y])))
        loss.backward()
        onehot = np.eye(10)[y]
        close(Z.grad.realize_cached_data(), (p - onehot) / 16)


def test_reshape_on_gpu():
    x = rng.standard_normal((4, 3, 2, 2)).astype(np.float32)
    X = Tensor(_gpu(x))
    out = reshape(X, (4, -1))
    assert out.shape == (4, 12)
    close(out.numpy(), x.reshape(4, -1))
    R = rng.standard_normal((4, 12))
    gx, = grads_of(out, R, [X])
    close(gx, R.reshape(x.shape))


def test_gpu_optimizers_match_numpy():
    from mnist_common import SGD, Adam
    for make in (lambda: SGD(0.1), lambda: Adam(0.01)):
        p_np = rng.standard_normal((5, 7)).astype(np.float32)
        p_gpu = _gpu(p_np.copy())
        opt_np, opt_gpu = make(), make()
        for _ in range(3):
            g = rng.standard_normal((5, 7)).astype(np.float32)
            opt_np.step([p_np], [g])
            opt_gpu.step([p_gpu], [_gpu(g)])
        close(p_gpu.numpy(), p_np, tol=1e-5)


def test_kaiming_uniform_init():
    from tiny_nn import kaiming_uniform
    fan_in = 784
    w = kaiming_uniform((784, 100), fan_in, seed=7).numpy()
    bound = np.sqrt(6.0 / fan_in)
    assert np.all(np.abs(w) <= bound + 1e-6)
    assert abs(w.mean()) < 0.01 * bound
    assert abs(w.std() - np.sqrt(2.0 / fan_in)) < 0.02 * np.sqrt(2.0 / fan_in)
    # 同一种子可复现，不同种子不同
    assert np.array_equal(w, kaiming_uniform((784, 100), fan_in, seed=7).numpy())
    assert not np.array_equal(w, kaiming_uniform((784, 100), fan_in, seed=8).numpy())
    # 不同 block 中线程号相同的位置不应重复（原实现每 256 个元素重复一次）
    flat = w.ravel()
    assert not np.allclose(flat[:256], flat[256:512])


def test_cnn_finite_difference():
    """整网梯度与中心差分对比（float32，容差放宽）"""
    model = CNN(seed=3)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 4)
    loss, _ = model.loss(x, y)
    loss.backward()
    def loss_with(w, data):
        w.cached_data = _gpu(data)
        return model.loss(x, y)[0].numpy().item()

    for k, w in enumerate(model.weights):
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
            w.cached_data = _gpu(data)
            numeric = (lp - lm) / (2 * eps)
            assert abs(numeric - g[idx]) <= 2e-2 * max(abs(numeric), 1e-2), \
                "param {} {}: analytic {:.5f} numeric {:.5f}".format(k, idx, g[idx], numeric)


def test_fused_ops_match_unfused():
    """融合算子（conv+ReLU+池化、FC+ReLU、单 kernel 交叉熵、一次更新全部参数的 SGD）与逐算子写法训练结果一致"""
    from mnist_common import SGD
    x = rng.standard_normal((40, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 40)
    for model_cls in (MLP, CNN):
        results = []
        for fused in (False, True):
            model, opt = model_cls(seed=5, use_graph=False, fused=fused), SGD(0.05)
            model.begin_epoch(x, y, np.arange(40))
            losses = [model.train_batch(i, 8, opt).numpy().item() for i in range(0, 40, 8)]
            results.append((losses, [w.numpy() for w in model.weights]))
        close(results[1][0], results[0][0], tol=1e-5)
        for a, b in zip(results[0][1], results[1][1]):
            close(b, a, tol=1e-5)


def test_conv_relu_maxpool():
    from operators import conv_relu_maxpool
    for shape, wshape, pad in [((2, 1, 8, 8), (3, 1, 3, 3), 1), ((3, 3, 7, 9), (10, 3, 3, 3), 1),
                               ((2, 2, 9, 8), (3, 2, 5, 5), 2)]:
        x = rng.standard_normal(shape).astype(np.float32)
        w = rng.standard_normal(wshape).astype(np.float32)
        R = rng.standard_normal((shape[0], wshape[0], (shape[2] + 2 * pad - wshape[2] + 1) // 2,
                                 (shape[3] + 2 * pad - wshape[3] + 1) // 2))
        outs = []
        for fused in (False, True):
            X, W = Tensor(x), Tensor(w)
            out = (conv_relu_maxpool(X, W, pad, pad) if fused else
                   maxpool(relu(conv(X, weight=W, pad_h=pad, pad_w=pad))))
            outs.append([out.realize_cached_data().numpy()] + grads_of(out, R, [X, W]))
        for a, b in zip(*outs):
            close(b, a)


def test_cuda_graph_matches_eager():
    """CUDA Graph 重放与逐算子执行的训练结果一致（SGD / Adam，MLP / CNN）"""
    from mnist_common import SGD, Adam
    x = rng.standard_normal((40, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 40)
    order, shuffled = np.arange(40), rng.permutation(40)
    for model_cls in (MLP, CNN):
        for make_opt in (lambda: SGD(0.05), lambda: Adam(1e-3)):
            results = []
            # 逐算子 / 每图 1 个 step / 每图 2 个 step（5 个 batch 时会同时用到 2 步图和 1 步图）
            for use_graph, chunk in ((False, 1), (True, 1), (True, 2)):
                model, opt = model_cls(seed=5, use_graph=use_graph, chunk=chunk), make_opt()
                for epoch in range(2):
                    model.begin_epoch(x, y, shuffled if epoch else order)
                    model.train_steps(0, 5, 8, opt)
                model.begin_epoch(x, y, order)
                model.train_steps(2, 3, 8, opt)
                model.train_batch(36, 4, opt)
                model.sync()
                results.append([w.numpy() for w in model.weights])
            for other in results[1:]:
                for eager, graph in zip(results[0], other):
                    close(graph, eager, tol=1e-5)


def test_training_step_keeps_graph_small():
    """参数更新后权重仍是叶子节点，不会把历史计算图串起来"""
    model = MLP(seed=0)
    from mnist_common import SGD
    x = rng.standard_normal((8, 1, 28, 28)).astype(np.float32)
    y = rng.integers(0, 10, 8)
    for _ in range(3):
        model.train_step(x, y, SGD(0.1))
    assert all(w.is_leaf() for w in model.weights)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
