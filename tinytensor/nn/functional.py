"""
无状态的网络算子（用法对应 torch.nn.functional）。每个函数是计算图上的一个节点，对应一到两个 CUDA kernel

一套接口，两组 kernel，由内存布局选择（与 PyTorch 的 memory_format 相同的思路）：
- contiguous_format（NCHW，默认）：myLayer 里的 fp32 kernel（课程部分），以及为小网络手写的融合 kernel
- channels_last（NHWC）        ：myNN 里的 kernel，fp32 或 bf16，卷积 / 全连接 / attention 用 tensor core

带参数的算子按参数的布局选择（model.to(memory_format=...) 决定），没有参数的算子按输入的布局选择；
输出继承这个布局标记，所以一个模型内部不会混用两组 kernel

约定：
- 参数（nn.Parameter）是计算图之外的叶子：算子反向时把参数梯度直接写进 Parameter，gradient() 只返回激活值的梯度
  （NCHW 的 kernel 每次写出一个新的梯度张量；NHWC 的 kernel 把梯度累加进参数的梯度缓冲，由 optimizer.zero_grad() 清零）
- 混合精度：输入是 bf16 时，卷积 / 全连接 / cls / pos 读优化器维护的 bf16 权重副本；BN / LN 的参数和统计量始终是 fp32
- 一个输入不需要梯度（例如输入图像）时不计算它的梯度（省掉第一层卷积的 dgrad）
"""
import numpy as np

from .. import _state
from .._C import myLayer as ml
from .._C import myNN
from ..tensor import Tensor, TensorOp, EWiseAdd, _gpu, _np, _grad, tf, ti, tb

ACTIVATIONS = {None: 0, "relu": 1, "gelu": 2}


def new_like(dtype):
    """输出张量的占位符，kernel 内部按需要的形状分配（显存来自缓存池）"""
    return tb([1], "gpu") if dtype == "bf16" else tf([1], "gpu")


def dtype_of(data):
    return "bf16" if isinstance(data, tb) else "fp32"


def const(data):
    return Tensor.make_const(data)


class NNOp(TensorOp):
    gpu = True
    has_params = False
    channels_last = None    # 输出的布局标记；None 表示沿用第一个输入的

    def __call__(self, *args):
        requires_grad = _state.grad_enabled and (self.has_params or any(a.requires_grad for a in args))
        out = Tensor.__new__(Tensor)
        if requires_grad:
            out._init(self, list(args), requires_grad=True)
            out.realize_cached_data()
        else:
            # 不需要梯度（no_grad 或输入、参数都不求导）：不保留计算图，中间结果用完即释放
            out._init(None, [], cached_data=self.compute(*[a.realize_cached_data() for a in args]),
                      requires_grad=False)
        if self.channels_last if self.channels_last is not None else args[0].channels_last:
            out.channels_last = True
        return out

    def release(self):
        """反向结束后释放算子保存的中间量"""


def _needs(node, i):
    return node.inputs[i].requires_grad


def _raw(out_grad):
    return out_grad.realize_cached_data()


def _on_gpu(data):
    """numpy 数组上传成 fp32；已经在 GPU 上的张量（fp32 / bf16）原样返回"""
    return _gpu(data) if isinstance(data, np.ndarray) else data


# =====================================================================================================
# contiguous_format（NCHW）：myLayer，fp32
# =====================================================================================================
class _Linear(NNOp):
    """y = x W + b（W 的物理形状是 [in, out]）；relu=True 时 bias 与 ReLU 在一个 kernel 里完成"""
    has_params, channels_last = True, False

    def __init__(self, weight, bias, relu):
        self.weight, self.bias, self.relu = weight, bias, relu

    def _bias(self):
        if self.bias is not None:
            return self.bias.raw
        return tf([self.weight.shape[1]], "gpu").zeros()

    def compute(self, x):
        x, w = _gpu(x), self.weight.raw
        if self.relu:
            out = tf([1], "gpu")
            ml.FcReluForward(x, out, w, self._bias())
        else:
            out = tf([x.get_shape()[0], w.get_shape()[1]], "gpu")
            ml.FcForward(x, out, w, self._bias())
        return out

    def gradient(self, out_grad, node):
        x, w, dy = _gpu(node.inputs[0].realize_cached_data()), self.weight.raw, _gpu(_raw(out_grad))
        need = _needs(node, 0)
        if self.relu:
            dx, dw, db = tf([1], "gpu"), tf([1], "gpu"), tf([1], "gpu")
            ml.FcReluBackward(x, node.realize_cached_data(), w, dx, dy, dw, db, need)
        else:
            b = self._bias()
            dw, db = tf(w.get_shape(), "gpu"), tf(b.get_shape(), "gpu")
            if need:
                dx = tf(x.get_shape(), "gpu")
                # 第二个参数（前向输出）在反向中用不到，传 dy 占位，避免多分配一块显存
                ml.FcBackward(x, dy, w, b, dx, dy, dw, db)
            else:
                # 输入不需要梯度（例如网络的输入数据）：只计算参数梯度
                ml.FcBackwardParams(x, w, b, dy, dw, db)
        self.weight.set_grad(dw)
        if self.bias is not None:
            self.bias.set_grad(db)
        return (_grad(dx) if need else None,)


class _Conv(NNOp):
    """卷积核的物理形状 [Cout, Cin, KH, KW]；Cin * KH * KW <= 256 时直接卷积，否则 im2col + GEMM"""
    has_params, channels_last = True, False

    def __init__(self, weight, stride, padding):
        self.weight, self.stride, self.padding = weight, stride, padding

    def compute(self, x):
        x, w, s, p = _gpu(x), self.weight.raw, self.stride, self.padding
        n, _, h, wd = x.get_shape()
        cout, _, kh, kw = w.get_shape()
        out = tf([n, cout, (h + 2 * p - kh) // s + 1, (wd + 2 * p - kw) // s + 1], "gpu")
        ml.ConvForward(x, out, w, p, p, s, s)
        return out

    def gradient(self, out_grad, node):
        x, w, dy = _gpu(node.inputs[0].realize_cached_data()), self.weight.raw, _gpu(_raw(out_grad))
        s, p = self.stride, self.padding
        dw = tf(w.get_shape(), "gpu")
        if _needs(node, 0):
            dx = tf(x.get_shape(), "gpu")
            ml.ConvBackward(x, dy, w, dx, dy, dw, p, p, s, s)
        else:
            # 输入不需要梯度（例如网络的输入图像）：只计算卷积核梯度
            dx = None
            ml.ConvBackwardWeight(x, w, dy, dw, p, p, s, s)
        self.weight.set_grad(dw)
        return (_grad(dx) if dx is not None else None,)


class _ConvReluMaxPool(NNOp):
    """
    maxpool2x2(relu(conv(x)))。三步在一个 kernel 里完成：卷积结果只留在寄存器里，不写回显存；
    反向由 argmax 直接还原卷积输出的梯度，不需要清零和 atomicAdd
    """
    has_params, channels_last = True, False

    def __init__(self, weight, stride, padding):
        self.weight, self.stride, self.padding = weight, stride, padding

    def compute(self, x):
        out, s, p = tf([1], "gpu"), self.stride, self.padding
        self.argmax = ti([1], "gpu")
        ml.ConvReluMaxPoolForward(_gpu(x), self.weight.raw, out, self.argmax, p, p, s, s)
        return out

    def gradient(self, out_grad, node):
        need, s, p = _needs(node, 0), self.stride, self.padding
        dx, dw = tf([1], "gpu"), tf([1], "gpu")
        ml.ConvReluMaxPoolBackward(_gpu(node.inputs[0].realize_cached_data()), self.weight.raw, _gpu(_raw(out_grad)),
                                   self.argmax, dx, dw, p, p, s, s, need)
        self.weight.set_grad(dw)
        return (_grad(dx) if need else None,)

    def release(self):
        self.argmax = None


class _MaxPool(NNOp):
    channels_last = False

    def __init__(self, kernel_size, stride, padding):
        self.kernel, self.stride, self.padding = [kernel_size, kernel_size], stride, padding

    def compute(self, x):
        x, s, p = _gpu(x), self.stride, self.padding
        n, c, h, w = x.get_shape()
        shape = [n, c, (h + 2 * p - self.kernel[0]) // s + 1, (w + 2 * p - self.kernel[1]) // s + 1]
        out, self.mask = tf(shape, "gpu"), tf(shape, "gpu")
        ml.MaxPoolingForward(x, out, self.mask, self.kernel, p, p, s, s)
        return out

    def gradient(self, out_grad, node):
        shape, s, p = list(node.inputs[0].shape), self.stride, self.padding
        dx = tf(shape, "gpu")
        ml.MaxPoolingBackward(_gpu(_raw(out_grad)), self.mask, dx, shape, self.kernel, p, p, s, s)
        return (_grad(dx),)

    def release(self):
        self.mask = None


class _ReLU(NNOp):
    channels_last = False

    def compute(self, x):
        return _gpu(x).ReluForward()

    def gradient(self, out_grad, node):
        return (_grad(_gpu(node.inputs[0].realize_cached_data()).ReluBackward(_gpu(_raw(out_grad)))),)


class _CrossEntropy(NNOp):
    """一个 kernel 同时算出 batch 平均 loss（GPU 上的 [1]，读取时才同步）和对 logits 的梯度 (softmax - onehot) / N"""
    channels_last = False

    def __init__(self, target):
        target = target.realize_cached_data() if isinstance(target, Tensor) else target
        self.target = target if isinstance(target, ti) else ti(np.ascontiguousarray(target, dtype=np.int32), "gpu")

    def compute(self, logits):
        loss, self.grad = tf([1], "gpu"), tf([1], "gpu")
        ml.SoftmaxCrossEntropy(_gpu(logits), self.target, loss, self.grad)
        return loss

    def gradient(self, out_grad, node):
        scale = float(np.asarray(_np(_raw(out_grad))).reshape(-1)[0])
        return (_grad(self.grad if scale == 1.0 else tf(self.grad).mults(scale)),)

    def release(self):
        self.grad = None


class _CrossEntropyReference(TensorOp):
    """numpy 实现的 softmax 交叉熵，返回 batch 平均 loss（测试用的参考实现）"""

    def __init__(self, target):
        self.target = _np(target.realize_cached_data() if isinstance(target, Tensor) else target).astype(np.int64)

    def compute(self, z):
        n = self.target.shape[0]
        z = z - np.max(z, axis=1, keepdims=True)
        log_prob = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
        self.softmax = np.exp(log_prob)
        return np.array(-np.mean(log_prob[np.arange(n), self.target]), dtype=np.float32)

    def gradient(self, out_grad, node):
        n = self.target.shape[0]
        onehot = np.zeros_like(self.softmax)
        onehot[np.arange(n), self.target] = 1
        return _grad((self.softmax - onehot) / n * out_grad.numpy())


# =====================================================================================================
# channels_last（NHWC）：myNN，fp32 / bf16
# =====================================================================================================
class _ConvCL(NNOp):
    """卷积核的物理形状 [Cout, KH, KW, Cin / groups]"""
    has_params, channels_last = True, True

    def __init__(self, weight, stride, padding, groups):
        self.weight, self.stride, self.padding, self.groups = weight, stride, padding, groups

    def compute(self, x):
        y, s, p = new_like(dtype_of(x)), self.stride, self.padding
        myNN.conv_forward(x, self.weight.storage(dtype_of(x)), y, p, p, s, s, self.groups)
        return y

    def gradient(self, out_grad, node):
        x, dy = node.inputs[0].realize_cached_data(), _raw(out_grad)
        dtype, s, p, g = dtype_of(x), self.stride, self.padding, self.groups
        myNN.conv_backward_weight(x, dy, self.weight.grad_buffer(), p, p, s, s, g, True)
        if not _needs(node, 0):
            return (None,)
        dx = new_like(dtype)
        myNN.conv_backward_data(dy, self.weight.storage(dtype), dx, x.get_shape(), p, p, s, s, g)
        return (const(dx),)


class _BatchNormCL(NNOp):
    """y = [relu](bn(x) [+ residual])；训练时用 batch 统计量并更新滑动平均，评估时用滑动平均"""
    has_params, channels_last = True, True

    def __init__(self, running_mean, running_var, weight, bias, training, momentum, eps, relu):
        self.running_mean, self.running_var, self.weight, self.bias = running_mean, running_var, weight, bias
        self.training, self.momentum, self.eps, self.relu = training, momentum, eps, relu

    def compute(self, x, residual=None):
        y = new_like(dtype_of(x))
        if self.training:
            self.stats = tf([1], "gpu")
            myNN.batchnorm_train(x, self.weight.raw, self.bias.raw, self.running_mean, self.running_var, self.stats, y,
                                 residual, self.relu, self.momentum, self.eps)
        else:
            myNN.batchnorm_eval(x, self.weight.raw, self.bias.raw, self.running_mean, self.running_var, y, residual,
                                self.relu, self.eps)
        return y

    def gradient(self, out_grad, node):
        x, y = node.inputs[0].realize_cached_data(), node.realize_cached_data()
        has_res = len(node.inputs) > 1
        dx = new_like(dtype_of(x))
        dres = new_like(dtype_of(x)) if has_res else None
        myNN.batchnorm_backward(x, y, _raw(out_grad), self.weight.raw, self.bias.raw, self.stats, dx,
                                self.weight.grad_buffer(), self.bias.grad_buffer(), dres, self.relu, self.eps)
        return (const(dx), const(dres)) if has_res else (const(dx),)

    def release(self):
        self.stats = None


class _MaxPool2x2CL(NNOp):
    channels_last = True

    def compute(self, x):
        y = new_like(dtype_of(x))
        self.argmax = myNN.Tensor_uint8([1], "gpu")
        myNN.maxpool2x2_forward(x, y, self.argmax)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.maxpool2x2_backward(_raw(out_grad), self.argmax, dx, x.get_shape())
        return (const(dx),)

    def release(self):
        self.argmax = None


class _GlobalAvgPoolCL(NNOp):
    channels_last = True

    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.avgpool_forward(x, y)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.avgpool_backward(_raw(out_grad), dx, x.get_shape())
        return (const(dx),)


class _LinearCL(NNOp):
    """y = act(x W^T + b)（W 的物理形状是 [out, in]）；out_f32=True 时 bf16 输入输出 fp32（分类头的 logits）"""
    has_params, channels_last = True, True

    def __init__(self, weight, bias, act, out_f32):
        self.weight, self.bias, self.act, self.out_f32 = weight, bias, act, out_f32

    def compute(self, x):
        dtype = dtype_of(x)
        y = tf([1], "gpu") if self.out_f32 else new_like(dtype)
        self.pre = new_like(dtype) if self.act == 2 else None
        myNN.linear_forward(x, self.weight.storage(dtype), self.bias.raw if self.bias is not None else None, y,
                            self.act, self.pre)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dtype = dtype_of(x)
        dx = new_like(dtype) if _needs(node, 0) else None
        myNN.linear_backward(x, self.weight.storage(dtype), node.realize_cached_data(), _raw(out_grad), self.act,
                             self.pre, dx, self.weight.grad_buffer(),
                             self.bias.grad_buffer() if self.bias is not None else None)
        return (const(dx) if dx is not None else None,)

    def release(self):
        self.pre = None


class _LayerNormCL(NNOp):
    has_params, channels_last = True, True

    def __init__(self, weight, bias, eps):
        self.weight, self.bias, self.eps = weight, bias, eps

    def compute(self, x):
        y = new_like(dtype_of(x))
        self.mean_rstd = tf([1], "gpu")
        myNN.layernorm_forward(x, self.weight.raw, self.bias.raw, y, self.mean_rstd, self.eps)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.layernorm_backward(x, _raw(out_grad), self.weight.raw, self.mean_rstd, dx, self.weight.grad_buffer(),
                                self.bias.grad_buffer())
        return (const(dx),)

    def release(self):
        self.mean_rstd = None


class _AttentionCL(NNOp):
    """输入 qkv [B, L, 3, H, hd]，输出 [B, L, H, hd]；softmax(QK^T / sqrt(hd)) V，整个 (b, h) 在一个 block 的 smem 内"""
    channels_last = True

    def __init__(self, heads):
        self.heads = heads

    def compute(self, qkv):
        out = new_like(dtype_of(qkv))
        self.probs = tf([1], "gpu")
        myNN.attention_forward(qkv, self.heads, out, self.probs)
        return out

    def gradient(self, out_grad, node):
        qkv = node.inputs[0].realize_cached_data()
        dqkv = new_like(dtype_of(qkv))
        myNN.attention_backward(qkv, self.probs, _raw(out_grad), self.heads, dqkv)
        return (const(dqkv),)

    def release(self):
        self.probs = None


class _TokensCL(NNOp):
    """[B, N, D] 的 patch 特征前面拼上 cls token，再加位置编码 -> [B, N+1, D]"""
    has_params, channels_last = True, True

    def __init__(self, cls_token, pos_embed):
        self.cls, self.pos = cls_token, pos_embed

    def compute(self, patches):
        dtype, out = dtype_of(patches), new_like(dtype_of(patches))
        myNN.tokens_forward(patches, self.cls.storage(dtype), self.pos.storage(dtype), out)
        return out

    def gradient(self, out_grad, node):
        dp = new_like(dtype_of(_raw(out_grad)))
        myNN.tokens_backward(_raw(out_grad), dp, self.cls.grad_buffer(), self.pos.grad_buffer())
        return (const(dp),)


class _SelectTokenCL(NNOp):
    """取一个 token（cls）：[B, L, D] -> [B, D]"""
    channels_last = True

    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.select_token_forward(x, y)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.select_token_backward(_raw(out_grad), dx, x.get_shape())
        return (const(dx),)


class _AddCL(NNOp):
    channels_last = True

    def compute(self, a, b):
        out = new_like(dtype_of(a))
        myNN.add(a, b, out)
        return out

    def gradient(self, out_grad, node):
        return out_grad, out_grad


class _ActCL(NNOp):
    """单独的激活 y = act(x)，act 1 relu / 2 gelu（融合版本把它并进 BN / Linear 的 kernel）"""
    channels_last = True

    def __init__(self, act):
        self.act = act

    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.act_forward(x, y, self.act)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.act_backward(x, _raw(out_grad), dx, self.act)
        return (const(dx),)


class _CrossEntropyCL(NNOp):
    """
    softmax 交叉熵（可带标签平滑与 Mixup / CutMix 的双标签）。一个 kernel 写出对 logits 的梯度（平均 loss 的梯度），
    并把这个 batch 的 loss 总和、预测正确数累加进 GPU 上的 loss_sum / correct —— 训练过程中不读回任何数值
    """
    channels_last = True

    def __init__(self, target, label_smoothing, buffers):
        self.target, self.label_smoothing = target, label_smoothing
        self.dlogits, self.loss_sum, self.correct = buffers

    def compute(self, logits):
        need_grad = _state.grad_enabled
        scale = 1.0 / logits.get_shape()[0] if need_grad else 1.0
        if isinstance(self.target, tuple):
            y, y2, lam = self.target
            myNN.softmax_ce_mix(logits, y, y2, lam, self.label_smoothing, self.dlogits, scale, self.loss_sum,
                                self.correct)
        else:
            myNN.softmax_ce(logits, self.target, self.dlogits, scale, self.loss_sum, self.correct, need_grad)
        return self.loss_sum

    def gradient(self, out_grad, node):
        return (const(self.dlogits),)


class _View(NNOp):
    """不拷贝的变形：输出与输入共享显存（view 会让输入的显存保持存活）"""

    def __init__(self, shape):
        self.shape = list(shape)

    def compute(self, x):
        x = _on_gpu(x)
        self.in_shape = x.get_shape()
        return x.view(0, self.shape)

    def gradient(self, out_grad, node):
        return (const(_on_gpu(_raw(out_grad)).view(0, self.in_shape)),)


# =====================================================================================================
# 接口
# =====================================================================================================
def _data(t):
    return t.realize_cached_data() if isinstance(t, Tensor) else t


def _contiguous_only(name, **unsupported):
    for key, (value, default) in unsupported.items():
        if value != default:
            raise NotImplementedError("{}: {}={!r} needs the channels_last kernels "
                                      "(model.to(memory_format=tinytensor.channels_last))".format(name, key, value))


def linear(input, weight, bias=None, activation=None, out_dtype=None):
    """
    全连接。activation="relu" / "gelu" 时激活融合在同一个算子里（nn.intrinsic.LinearReLU / LinearGELU 用它）；
    out_dtype=float32 让 bf16 输入的层输出 fp32（分类头的 logits，只有 channels_last 支持）
    """
    if weight.channels_last:
        return _LinearCL(weight, bias, ACTIVATIONS[activation], out_dtype == _state.float32)(input)
    _contiguous_only("linear", out_dtype=(out_dtype, None))
    if activation not in (None, "relu"):
        raise NotImplementedError("linear: activation={!r} needs the channels_last kernels".format(activation))
    return _Linear(weight, bias, activation == "relu")(input)


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    if bias is not None:
        raise NotImplementedError("conv2d: the kernels have no bias (follow the convolution with batch_norm)")
    if weight.channels_last:
        return _ConvCL(weight, stride, padding, groups)(input)
    _contiguous_only("conv2d", groups=(groups, 1))
    return _Conv(weight, stride, padding)(input)


def conv_relu_max_pool2d(input, weight, stride=1, padding=0):
    """
    max_pool2d(relu(conv2d(x)), 2) 的融合版本（contiguous_format）。
    通道数较大（Cin * KH * KW > 256）时退回三个独立算子：那时卷积走 im2col + GEMM，融合 kernel 没有优势
    """
    if weight.channels_last:
        raise NotImplementedError("conv_relu_max_pool2d is a contiguous_format kernel; "
                                  "channels_last models fuse BN + ReLU instead (nn.intrinsic.ConvBNAct2d)")
    _, c_in, kh, kw = weight.shape
    if c_in * kh * kw > 256:
        return max_pool2d(relu(conv2d(input, weight, None, stride, padding)), 2)
    return _ConvReluMaxPool(weight, stride, padding)(input)


def relu(input):
    return _ActCL(1)(input) if input.channels_last else _ReLU()(input)


def gelu(input):
    if not input.channels_last:
        raise NotImplementedError("gelu needs the channels_last kernels")
    return _ActCL(2)(input)


def max_pool2d(input, kernel_size=2, stride=None, padding=0):
    stride = kernel_size if stride is None else stride
    if input.channels_last:
        if (kernel_size, stride, padding) != (2, 2, 0):
            raise NotImplementedError("channels_last max_pool2d supports only a 2x2 window with stride 2")
        return _MaxPool2x2CL()(input)
    return _MaxPool(kernel_size, stride, padding)(input)


def global_avg_pool2d(input):
    """[B, H, W, C] -> [B, C]"""
    if not input.channels_last:
        raise NotImplementedError("global_avg_pool2d needs the channels_last kernels")
    return _GlobalAvgPoolCL()(input)


def batch_norm(input, running_mean, running_var, weight, bias, training=True, momentum=0.1, eps=1e-5, relu=False,
               residual=None):
    """
    BatchNorm（channels_last）。relu / residual 给出时计算 [relu](bn(x) [+ residual])，都在同一个 kernel 里
    （nn.intrinsic.ConvBNAct2d 用它）；running_mean / running_var 是 GPU 张量，训练时原地更新
    """
    op = _BatchNormCL(_data(running_mean), _data(running_var), weight, bias, training, momentum, eps, relu)
    return op(input) if residual is None else op(input, residual)


def layer_norm(input, weight, bias, eps=1e-6):
    """对最后一维做 LayerNorm（channels_last）"""
    return _LayerNormCL(weight, bias, eps)(input)


def self_attention(qkv, heads):
    """多头自注意力的核心：qkv [B, L, 3, heads, head_dim] -> [B, L, heads, head_dim]"""
    return _AttentionCL(heads)(qkv)


def embed_tokens(patches, cls_token, pos_embed):
    """ViT 的输入：[B, N, D] 的 patch 特征前面拼上 cls token，再加位置编码 -> [B, N + 1, D]"""
    return _TokensCL(cls_token, pos_embed)(patches)


def select_token(input):
    """取第 0 个 token（cls）：[B, L, D] -> [B, D]"""
    return _SelectTokenCL()(input)


def add(a, b):
    return _AddCL()(a, b) if a.channels_last else EWiseAdd()(a, b)


def view(input, shape):
    """不拷贝的变形（形状里可以有一个 -1）"""
    shape = list(shape)
    if -1 in shape:
        known = int(np.prod([s for s in shape if s != -1]))
        shape[shape.index(-1)] = int(np.prod(input.shape)) // known
    return _View(shape)(input)


def flatten(input, start_dim=1):
    shape = list(input.shape)
    if len(shape) <= start_dim + 1:
        return input
    return view(input, shape[:start_dim] + [int(np.prod(shape[start_dim:]))])


def cross_entropy(input, target, label_smoothing=0.0, buffers=None):
    """
    softmax 交叉熵，梯度是 batch 平均 loss 对 logits 的梯度。

    contiguous_format : 返回 batch 平均 loss（GPU 上的 [1]）
    channels_last     : 把这个 batch 的 loss 总和与预测正确数累加进 buffers = (dlogits, loss_sum, correct)，
                        返回的张量就是累加器 loss_sum（nn.CrossEntropyLoss 持有这组缓冲，用 statistics() 读取并清零）。
                        target 可以是 (y, y2, lam)：Mixup / CutMix 的两组标签与混合比例
    """
    if not input.channels_last:
        if label_smoothing or isinstance(target, tuple):
            raise NotImplementedError("label smoothing / mixed targets need the channels_last kernels")
        return _CrossEntropy(target)(input)
    if buffers is None:
        buffers = (tf([1], "gpu"), tf([1], "gpu").zeros(), ti([1], "gpu").zeros())
    if isinstance(target, tuple):
        target = tuple(_data(t) for t in target)
    else:
        target = _data(target)
        if label_smoothing:
            # 标签平滑的 kernel 是双标签版本：第二个标签取同一个，混合比例 1
            target = (target, target, tf(np.ones(1, np.float32), "gpu"))
    return _CrossEntropyCL(target, label_smoothing, buffers)(input)


def cross_entropy_reference(input, target):
    """numpy 参考实现（测试用）"""
    return _CrossEntropyReference(target)(input)
