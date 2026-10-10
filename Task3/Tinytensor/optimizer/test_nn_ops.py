"""
myNN 算子与 PyTorch 逐一对照（前向 + 反向；fp32 与 bf16 两种精度）
运行：python test_nn_ops.py（需要 GPU 与 PyTorch）
"""
import numpy as np
import torch
import torch.nn.functional as F

import myNN as nn
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rng = np.random.default_rng(0)
DTYPES = ("fp32", "bf16")


def to_tt(a, dtype):
    t = tf(np.ascontiguousarray(a, dtype=np.float32), "gpu")
    return nn.to_bf16(t) if dtype == "bf16" else t


def empty(dtype):
    return nn.Tensor_bf16([1], "gpu") if dtype == "bf16" else tf([1], "gpu")


def zeros_f(shape):
    return tf(list(shape), "gpu").zeros()


def rounded(a, dtype):
    """bf16 时先把参考输入也舍入到 bf16，排除输入量化带来的差异"""
    if dtype == "bf16":
        return torch.tensor(a).to(torch.bfloat16).float().numpy()
    return np.asarray(a, dtype=np.float32)


def check(name, got, want, dtype, tol=None):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    scale = np.abs(want).max() + 1e-6
    err = np.abs(got - want).max() / scale
    tol = tol or (2e-5 if dtype == "fp32" else 2e-2)
    assert err < tol, "{} [{}]: rel err {:.2e} > {:.0e}".format(name, dtype, err, tol)


def tt_t(a):
    return torch.tensor(a, dtype=torch.float64, requires_grad=True)


def test_conv():
    cases = [  # (N, H, W, C, K, R, pad, stride, groups)
        (2, 9, 9, 8, 16, 3, 1, 1, 1), (2, 9, 9, 8, 16, 3, 1, 2, 1), (2, 8, 8, 16, 8, 1, 0, 2, 1),
        (2, 7, 7, 3, 16, 3, 1, 1, 1), (2, 8, 8, 32, 32, 3, 1, 1, 8), (2, 8, 8, 32, 64, 3, 1, 2, 4),
        (2, 16, 16, 3, 8, 4, 0, 4, 1),
        # 3 通道输入的第一层（conv_stem.cu）：宽度跨 64 的 tile 边界
        (2, 9, 70, 3, 64, 3, 1, 1, 1), (2, 5, 5, 3, 128, 3, 0, 1, 1),
        # 通道对齐：bf16 时走 tensor core 版本（多个 tile、步长 2、1x1、分组）
        (2, 9, 9, 32, 64, 3, 1, 1, 1), (2, 9, 9, 64, 32, 3, 1, 2, 1), (2, 8, 8, 64, 128, 1, 0, 2, 1),
        (2, 16, 16, 128, 256, 3, 1, 1, 1), (3, 7, 7, 64, 64, 3, 1, 1, 1), (2, 8, 8, 256, 256, 3, 1, 1, 8),
        (2, 12, 12, 64, 200, 3, 1, 1, 1), (2, 10, 10, 64, 64, 3, 1, 2, 1), (2, 8, 8, 32, 64, 1, 0, 2, 1),
        # 每组 4 / 8 / 16 通道的分组卷积（conv_grouped.cu），含步长 2
        (2, 9, 9, 128, 128, 3, 1, 1, 32), (2, 9, 9, 256, 256, 3, 1, 2, 32), (2, 6, 6, 512, 512, 3, 1, 1, 32),
        (2, 10, 10, 512, 512, 3, 1, 2, 32)]
    for dtype in DTYPES:
        for N, H, W, C, K, R, pad, stride, groups in cases:
            x = rounded(rng.standard_normal((N, H, W, C)), dtype)
            w = rounded(rng.standard_normal((K, R, R, C // groups)) * 0.2, dtype)
            xt, wt = tt_t(x), tt_t(w)
            yt = F.conv2d(xt.permute(0, 3, 1, 2), wt.permute(0, 3, 1, 2), padding=pad, stride=stride, groups=groups)
            yt = yt.permute(0, 2, 3, 1)
            dy = rounded(rng.standard_normal(yt.shape), dtype)
            yt.backward(torch.tensor(dy, dtype=torch.float64))
            X, Wt = to_tt(x, dtype), to_tt(w, dtype)
            y = empty(dtype)
            nn.conv_forward(X, Wt, y, pad, pad, stride, stride, groups)
            tag = "conv{}".format((N, H, W, C, K, R, pad, stride, groups))
            check(tag + " fwd", y.numpy(), yt.detach().numpy(), dtype)
            dx = empty(dtype)
            nn.conv_backward_data(to_tt(dy, dtype), Wt, dx, [N, H, W, C], pad, pad, stride, stride, groups)
            check(tag + " dgrad", dx.numpy(), xt.grad.numpy(), dtype)
            dw = zeros_f(w.shape)
            nn.conv_backward_weight(X, to_tt(dy, dtype), dw, pad, pad, stride, stride, groups, True)
            check(tag + " wgrad", dw.numpy(), wt.grad.numpy(), dtype)


def test_batchnorm():
    # C=24 走标量 kernel；C=64 / 512 走向量化 kernel（8 个通道一组，C/8 整除 256）
    for dtype in DTYPES:
        for M, C in (((4, 6, 6), 24), ((4, 9, 9), 64), ((2, 5, 7), 512)):
            for res, relu in ((False, True), (True, True), (False, False), (True, False)):
                x = rounded(rng.standard_normal(M + (C,)) * 2 + 0.5, dtype)
                r = rounded(rng.standard_normal(M + (C,)), dtype)
                gamma = rng.uniform(0.5, 1.5, C).astype(np.float32)
                beta = rng.standard_normal(C).astype(np.float32)
                xt, rt, gt, bt = tt_t(x), tt_t(r), tt_t(gamma), tt_t(beta)
                rm, rv = torch.zeros(C, dtype=torch.float64), torch.ones(C, dtype=torch.float64)
                yt = F.batch_norm(xt.reshape(-1, C), rm, rv, gt, bt, training=True, momentum=0.1, eps=1e-5).reshape(xt.shape)
                if res:
                    yt = yt + rt
                if relu:
                    yt = torch.relu(yt)
                dy = rounded(rng.standard_normal(yt.shape), dtype)
                yt.backward(torch.tensor(dy, dtype=torch.float64))

                X = to_tt(x, dtype)
                R = to_tt(r, dtype) if res else None
                G, B = tf(gamma, "gpu"), tf(beta, "gpu")
                run_m, run_v, stats = zeros_f([C]), tf(np.ones(C, np.float32), "gpu"), tf([1], "gpu")
                y = empty(dtype)
                nn.batchnorm_train(X, G, B, run_m, run_v, stats, y, R, relu, 0.1, 1e-5)
                tag = "bn(C={}, res={}, relu={})".format(C, res, relu)
                check(tag + " fwd", y.numpy(), yt.detach().numpy(), dtype)
                check(tag + " running_mean", run_m.numpy(), rm.numpy(), dtype)
                check(tag + " running_var", run_v.numpy(), rv.numpy(), dtype)
                dx, dg, db = empty(dtype), zeros_f([C]), zeros_f([C])
                dres = empty(dtype) if res else None
                nn.batchnorm_backward(X, y, to_tt(dy, dtype), G, stats, dx, dg, db, dres, relu, 1e-5)
                check(tag + " dx", dx.numpy(), xt.grad.numpy(), dtype, 5e-5 if dtype == "fp32" else 4e-2)
                check(tag + " dgamma", dg.numpy(), gt.grad.numpy(), dtype)
                check(tag + " dbeta", db.numpy(), bt.grad.numpy(), dtype)
                if res:
                    check(tag + " dres", dres.numpy(), rt.grad.numpy(), dtype)
                ye = empty(dtype)
                nn.batchnorm_eval(X, G, B, run_m, run_v, ye, R, relu, 1e-5)
                want = F.batch_norm(torch.tensor(x).reshape(-1, C), torch.tensor(run_m.numpy()), torch.tensor(run_v.numpy()),
                                    torch.tensor(gamma), torch.tensor(beta), training=False, eps=1e-5).reshape(x.shape)
                if res:
                    want = want + torch.tensor(r)
                if relu:
                    want = torch.relu(want)
                check(tag + " eval", ye.numpy(), want.numpy(), dtype)


def test_act():
    # 单独的 ReLU / GELU（--no-fuse 时用），对照 PyTorch
    for dtype in DTYPES:
        for act, fn in ((1, torch.relu), (2, F.gelu)):
            x = rounded(rng.standard_normal((3, 5, 7, 16)) * 2, dtype)
            dy = rounded(rng.standard_normal(x.shape), dtype)
            xt = tt_t(x)
            yt = fn(xt)
            yt.backward(torch.tensor(dy, dtype=torch.float64))
            X, y, dx = to_tt(x, dtype), empty(dtype), empty(dtype)
            nn.act_forward(X, y, act)
            nn.act_backward(X, to_tt(dy, dtype), dx, act)
            check("act{} fwd".format(act), y.numpy(), yt.detach().numpy(), dtype)
            check("act{} bwd".format(act), dx.numpy(), xt.grad.numpy(), dtype)


def test_layernorm():
    for dtype in DTYPES:
        x = rounded(rng.standard_normal((3, 5, 48)) * 3 + 1, dtype)
        gamma = rng.uniform(0.5, 1.5, 48).astype(np.float32)
        beta = rng.standard_normal(48).astype(np.float32)
        xt, gt, bt = tt_t(x), tt_t(gamma), tt_t(beta)
        yt = F.layer_norm(xt, (48,), gt, bt, eps=1e-6)
        dy = rounded(rng.standard_normal(yt.shape), dtype)
        yt.backward(torch.tensor(dy, dtype=torch.float64))
        X, G, B = to_tt(x, dtype), tf(gamma, "gpu"), tf(beta, "gpu")
        y, mr = empty(dtype), tf([1], "gpu")
        nn.layernorm_forward(X, G, B, y, mr, 1e-6)
        check("layernorm fwd", y.numpy(), yt.detach().numpy(), dtype)
        dx, dg, db = empty(dtype), zeros_f([48]), zeros_f([48])
        nn.layernorm_backward(X, to_tt(dy, dtype), G, mr, dx, dg, db)
        check("layernorm dx", dx.numpy(), xt.grad.numpy(), dtype, 5e-5 if dtype == "fp32" else 4e-2)
        check("layernorm dgamma", dg.numpy(), gt.grad.numpy(), dtype)
        check("layernorm dbeta", db.numpy(), bt.grad.numpy(), dtype)


def test_pooling():
    for dtype in DTYPES:
        x = rounded(rng.standard_normal((2, 8, 6, 5)), dtype)
        xt = tt_t(x)
        yt = F.max_pool2d(xt.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        dy = rounded(rng.standard_normal(yt.shape), dtype)
        yt.backward(torch.tensor(dy, dtype=torch.float64))
        X = to_tt(x, dtype)
        y, arg = empty(dtype), nn.Tensor_uint8([1], "gpu")
        nn.maxpool2x2_forward(X, y, arg)
        check("maxpool fwd", y.numpy(), yt.detach().numpy(), dtype)
        dx = empty(dtype)
        nn.maxpool2x2_backward(to_tt(dy, dtype), arg, dx, list(x.shape))
        check("maxpool bwd", dx.numpy(), xt.grad.numpy(), dtype)

        xt = tt_t(x)
        yt = xt.mean(dim=(1, 2))
        dy = rounded(rng.standard_normal(yt.shape), dtype)
        yt.backward(torch.tensor(dy, dtype=torch.float64))
        y = empty(dtype)
        nn.avgpool_forward(X, y)
        check("avgpool fwd", y.numpy(), yt.detach().numpy(), dtype)
        dx = empty(dtype)
        nn.avgpool_backward(to_tt(dy, dtype), dx, list(x.shape))
        check("avgpool bwd", dx.numpy(), xt.grad.numpy(), dtype)


def test_linear():
    for dtype in DTYPES:
        for act in (0, 1, 2):
            for out_f32 in ((False, True) if dtype == "bf16" else (False,)):
                if out_f32 and act == 2:
                    continue
                x = rounded(rng.standard_normal((2, 7, 24)), dtype)
                w = rounded(rng.standard_normal((40, 24)) * 0.2, dtype)
                b = rng.standard_normal(40).astype(np.float32)
                xt, wt, bt = tt_t(x), tt_t(w), tt_t(b)
                yt = xt @ wt.T + bt
                yt = torch.relu(yt) if act == 1 else F.gelu(yt) if act == 2 else yt
                dy = rounded(rng.standard_normal(yt.shape), dtype if not out_f32 else "fp32")
                yt.backward(torch.tensor(dy, dtype=torch.float64))
                X, Wt, Bt = to_tt(x, dtype), to_tt(w, dtype), tf(b, "gpu")
                y = tf([1], "gpu") if out_f32 else empty(dtype)
                pre = empty(dtype) if act == 2 else None
                nn.linear_forward(X, Wt, Bt, y, act, pre)
                tag = "linear(act={}, f32out={})".format(act, out_f32)
                check(tag + " fwd", y.numpy(), yt.detach().numpy(), dtype)
                dx, dw, db = empty(dtype), zeros_f(w.shape), zeros_f([40])
                DY = tf(dy.astype(np.float32), "gpu") if out_f32 else to_tt(dy, dtype)
                nn.linear_backward(X, Wt, y, DY, act, pre, dx, dw, db)
                check(tag + " dx", dx.numpy(), xt.grad.numpy(), dtype)
                check(tag + " dw", dw.numpy(), wt.grad.numpy(), dtype)
                check(tag + " db", db.numpy(), bt.grad.numpy(), dtype)


def test_attention():
    for dtype in DTYPES:
        for B, L, H, hd in ((2, 13, 3, 16), (2, 65, 3, 64)):   # 后者是 ViT-Tiny 的实际大小（序列跨 64）
            qkv = rounded(rng.standard_normal((B, L, 3, H, hd)), dtype)
            qt = tt_t(qkv)
            q, k, v = qt[:, :, 0].transpose(1, 2), qt[:, :, 1].transpose(1, 2), qt[:, :, 2].transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)   # [B, L, H, hd]
            do = rounded(rng.standard_normal(o.shape), dtype)
            o.backward(torch.tensor(do, dtype=torch.float64))
            Q = to_tt(qkv, dtype)
            out, probs = empty(dtype), tf([1], "gpu")
            nn.attention_forward(Q, H, out, probs)
            check("attention fwd", out.numpy(), o.detach().numpy(), dtype)
            dq = empty(dtype)
            nn.attention_backward(Q, probs, to_tt(do, dtype), H, dq)
            check("attention bwd", dq.numpy(), qt.grad.numpy(), dtype)


def test_tokens():
    for dtype in DTYPES:
        B, Np, D = 3, 4, 8
        p = rounded(rng.standard_normal((B, Np, D)), dtype)
        cls = rounded(rng.standard_normal(D), dtype)
        pos = rounded(rng.standard_normal((Np + 1, D)), dtype)
        pt, ct, st = tt_t(p), tt_t(cls), tt_t(pos)
        tok = torch.cat([ct.expand(B, 1, D), pt], 1) + st
        sel = tok[:, 0]
        dsel = rounded(rng.standard_normal(sel.shape), dtype)
        dtok_extra = rounded(rng.standard_normal(tok.shape), dtype)
        (sel * torch.tensor(dsel)).sum().backward(retain_graph=True)
        tok.backward(torch.tensor(dtok_extra, dtype=torch.float64))
        out = empty(dtype)
        nn.tokens_forward(to_tt(p, dtype), to_tt(cls, dtype), to_tt(pos, dtype), out)
        check("tokens fwd", out.numpy(), tok.detach().numpy(), dtype)
        s = empty(dtype)
        nn.select_token_forward(out, s)
        check("select fwd", s.numpy(), sel.detach().numpy(), dtype)
        dtok = empty(dtype)
        nn.select_token_backward(to_tt(dsel, dtype), dtok, [B, Np + 1, D])
        total = empty(dtype)
        nn.add(dtok, to_tt(dtok_extra, dtype), total)
        dp, dc, dpos = empty(dtype), zeros_f([D]), zeros_f([Np + 1, D])
        nn.tokens_backward(total, dp, dc, dpos)
        check("tokens dpatch", dp.numpy(), pt.grad.numpy(), dtype)
        check("tokens dcls", dc.numpy(), ct.grad.numpy(), dtype)
        check("tokens dpos", dpos.numpy(), st.grad.numpy(), dtype)


def test_softmax_ce():
    z = rng.standard_normal((37, 200)).astype(np.float32) * 3
    y = rng.integers(0, 200, 37).astype(np.int32)
    zt = tt_t(z)
    loss = F.cross_entropy(zt, torch.tensor(y, dtype=torch.long), reduction="mean")
    loss.backward()
    grad, ls, corr = tf([1], "gpu"), zeros_f([1]), ti([1], "gpu").zeros()
    nn.softmax_ce(tf(z, "gpu"), ti(y, "gpu"), grad, 1.0 / 37, ls, corr, True)
    check("ce loss", ls.numpy()[0] / 37, loss.item(), "fp32")
    check("ce grad", grad.numpy(), zt.grad.numpy(), "fp32")
    assert corr.numpy()[0] == int((z.argmax(1) == y).sum())


def hash32(x):
    x = np.uint32(x)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint32(16); x *= np.uint32(0x7feb352d)
        x ^= x >> np.uint32(15); x *= np.uint32(0x846ca68b)
        x ^= x >> np.uint32(16)
    return x


def reference_batch(images, labels, order, base, t, batch, mean, std, augment, pad, seed):
    N, H, W, _ = images.shape
    out = np.zeros((batch, H, W, 3), np.float32)
    for b in range(batch):
        idx = order[base + b]
        dy = dx = flip = 0
        if augment:
            with np.errstate(over="ignore"):
                rnd = int(hash32(np.uint32(seed) ^ hash32(np.uint32(t) * np.uint32(0x9E3779B9) + np.uint32(b))))
            rng_ = 2 * pad + 1
            dy, dx, flip = rnd % rng_ - pad, (rnd >> 8) % rng_ - pad, (rnd >> 16) & 1
        # 输出 (h, w) 取源像素 (h + dy, (flip ? W-1-w : w) + dx)，越界补 0（归一化之前）
        sh = np.arange(H) + dy
        sw = (W - 1 - np.arange(W) if flip else np.arange(W)) + dx
        valid = ((sh >= 0) & (sh < H))[:, None] & ((sw >= 0) & (sw < W))[None, :]
        crop = images[idx][np.clip(sh, 0, H - 1)][:, np.clip(sw, 0, W - 1)] / 255.0
        crop = np.where(valid[..., None], crop, 0.0)
        out[b] = (crop - mean) / std
    return out, labels[order[base:base + batch]]


def test_load_batch():
    images = rng.integers(0, 256, (20, 8, 8, 3)).astype(np.uint8)
    labels = rng.integers(0, 200, 20).astype(np.int32)
    order = rng.permutation(20).astype(np.int32)
    mean, std = [0.48, 0.45, 0.40], [0.28, 0.27, 0.28]
    for dtype in DTYPES:
        for augment in (False, True):
            step = ti(np.array([1, 0, 5, 0], np.int32), "gpu")
            x = nn.Tensor_bf16([6, 8, 8, 3], "gpu") if dtype == "bf16" else tf([6, 8, 8, 3], "gpu")
            y = ti([6], "gpu")
            nn.load_batch(nn.Tensor_uint8(images, "gpu"), ti(labels, "gpu"), ti(order, "gpu"), step, x, y, 6,
                          mean, std, augment, 2, 1234)
            want_x, want_y = reference_batch(images, labels, order, 6, 5, 6, np.array(mean), np.array(std), augment, 2, 1234)
            check("load_batch(aug={})".format(augment), x.numpy(), want_x, dtype, 1e-5 if dtype == "fp32" else 1e-2)
            assert (y.numpy() == want_y).all()
            assert list(step.numpy()) == [2, 0, 6, 0], step.numpy()


def test_optimizers():
    n = 1000
    w0 = rng.standard_normal(n).astype(np.float32)
    grads = [rng.standard_normal(n).astype(np.float32) for _ in range(5)]
    sched = nn.LrSchedule(0.1, 2, 5, 0.1)

    def lr_at(t):
        if t < 2:
            return 0.1 * (t + 1) / 2
        p = min(1.0, (t - 2) / 3)
        return 0.1 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * p)))

    for name in ("sgd", "adamw"):
        p = torch.tensor(w0.astype(np.float64), requires_grad=True)
        opt = (torch.optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) if name == "sgd"
               else torch.optim.AdamW([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05))
        w, m, v = tf(w0, "gpu"), zeros_f([n]), zeros_f([n])
        wl = nn.Tensor_bf16([n], "gpu")
        step = ti(np.array([0, 0, 0, 0], np.int32), "gpu")
        for t, g in enumerate(grads):
            for group in opt.param_groups:
                group["lr"] = lr_at(t)
            p.grad = torch.tensor(g.astype(np.float64))
            opt.step()
            step.copy_from(ti(np.array([0, 0, t + 1, 0], np.int32), "gpu"), 0)
            if name == "sgd":
                nn.sgd_step(w, tf(g, "gpu"), m, wl, step, sched, 0.9, 5e-4, True)
            else:
                nn.adamw_step(w, tf(g, "gpu"), m, v, wl, step, sched, 0.9, 0.999, 1e-8, 0.05)
        check(name, w.numpy(), p.detach().numpy(), "fp32", 1e-5)
        check(name + " bf16 copy", wl.numpy(), p.detach().numpy(), "bf16", 1e-2)


if __name__ == "__main__":
    for fn in [test_conv, test_batchnorm, test_act, test_layernorm, test_pooling, test_linear, test_attention, test_tokens,
               test_softmax_ce, test_load_batch, test_optimizers]:
        fn()
        print("PASS", fn.__name__)
