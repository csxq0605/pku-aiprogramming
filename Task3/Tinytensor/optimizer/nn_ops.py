"""
图像模型（VGG / ResNet / ResNeXt / ViT）用到的自动微分算子，底层调用 myNN（NHWC，fp32 或 bf16）

约定：
- 计算图只连接激活值；参数（Param）作为算子的属性传入。反向时参数梯度直接累加进扁平的 fp32 梯度缓冲
  （每个 step 开始时整体清零一次），gradient() 只返回激活值的梯度
- 混合精度：卷积 / 全连接 / cls / pos 用优化器维护的 bf16 权重副本；BN / LN 的参数和统计量始终是 fp32
- 一个输入不需要梯度（例如输入图像）时不计算它的梯度（省掉第一层卷积的 dgrad）
"""
import numpy as np

import myNN
from myTensor import Tensor_float as tf
from operators import Tensor, TensorOp

tb = myNN.Tensor_bf16


def new_like(dtype):
    """输出张量的占位符，算子内部按需要的形状分配（显存来自缓存池）"""
    return tb([1], "gpu") if dtype == "bf16" else tf([1], "gpu")


def dtype_of(data):
    return "bf16" if isinstance(data, tb) else "fp32"


def const(data):
    return Tensor.make_const(data)


class Param:
    """一个参数：fp32 主权重、fp32 梯度、bf16 副本都是扁平缓冲上的视图（不拥有显存）"""

    def __init__(self, name, shape, init, fan_in=None):
        self.name, self.shape, self.init, self.fan_in = name, tuple(shape), init, fan_in
        self.w = self.g = self.lowp = None

    def weight(self, dtype):
        return self.lowp if dtype == "bf16" else self.w


class NNOp(TensorOp):
    gpu = True
    has_params = False

    def __call__(self, *args):
        tensor = Tensor.__new__(Tensor)
        requires_grad = self.has_params or any(a.requires_grad for a in args)
        tensor._init(self, list(args), requires_grad=requires_grad)
        tensor.realize_cached_data()
        return tensor

    def release(self):
        """反向结束后释放算子保存的中间量"""
        self.saved = None


def _needs(node, i):
    return node.inputs[i].requires_grad


class Conv(NNOp):
    has_params = True

    def __init__(self, p, stride=1, pad=0, groups=1):
        self.p, self.stride, self.pad, self.groups = p, stride, pad, groups

    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.conv_forward(x, self.p.weight(dtype_of(x)), y, self.pad, self.pad, self.stride, self.stride, self.groups)
        return y

    def gradient(self, out_grad, node):
        x, dy = node.inputs[0].realize_cached_data(), out_grad.realize_cached_data()
        dtype, s, p, g = dtype_of(x), self.stride, self.pad, self.groups
        myNN.conv_backward_weight(x, dy, self.p.g, p, p, s, s, g, True)
        if not _needs(node, 0):
            return (None,)
        dx = new_like(dtype)
        myNN.conv_backward_data(dy, self.p.weight(dtype), dx, x.get_shape(), p, p, s, s, g)
        return (const(dx),)


class BatchNorm(NNOp):
    """y = [relu](bn(x) [+ residual])；训练时用 batch 统计量并更新滑动平均，评估时用滑动平均"""
    has_params = True

    def __init__(self, bn, relu, training):
        self.bn, self.relu, self.training = bn, relu, training

    def compute(self, x, residual=None):
        bn, y = self.bn, new_like(dtype_of(x))
        if self.training:
            self.stats = tf([1], "gpu")
            myNN.batchnorm_train(x, bn.gamma.w, bn.beta.w, bn.running_mean, bn.running_var, self.stats, y, residual,
                                 self.relu, bn.momentum, bn.eps)
        else:
            myNN.batchnorm_eval(x, bn.gamma.w, bn.beta.w, bn.running_mean, bn.running_var, y, residual, self.relu,
                                bn.eps)
        return y

    def gradient(self, out_grad, node):
        bn = self.bn
        x, y = node.inputs[0].realize_cached_data(), node.realize_cached_data()
        has_res = len(node.inputs) > 1
        dx = new_like(dtype_of(x))
        dres = new_like(dtype_of(x)) if has_res else None
        myNN.batchnorm_backward(x, y, out_grad.realize_cached_data(), bn.gamma.w, self.stats, dx, bn.gamma.g,
                                bn.beta.g, dres, self.relu, bn.eps)
        return (const(dx), const(dres)) if has_res else (const(dx),)

    def release(self):
        self.stats = None


class MaxPool2x2(NNOp):
    def compute(self, x):
        y = new_like(dtype_of(x))
        self.argmax = myNN.Tensor_uint8([1], "gpu")
        myNN.maxpool2x2_forward(x, y, self.argmax)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.maxpool2x2_backward(out_grad.realize_cached_data(), self.argmax, dx, x.get_shape())
        return (const(dx),)

    def release(self):
        self.argmax = None


class GlobalAvgPool(NNOp):
    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.avgpool_forward(x, y)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.avgpool_backward(out_grad.realize_cached_data(), dx, x.get_shape())
        return (const(dx),)


class Linear(NNOp):
    """y = act(x W^T + b)，act: 0 无 / 1 relu / 2 gelu；out_f32=True 时 bf16 输入输出 fp32（分类头的 logits）"""
    has_params = True

    def __init__(self, w, b, act=0, out_f32=False):
        self.w, self.b, self.act, self.out_f32 = w, b, act, out_f32

    def compute(self, x):
        dtype = dtype_of(x)
        y = tf([1], "gpu") if self.out_f32 else new_like(dtype)
        self.pre = new_like(dtype) if self.act == 2 else None
        myNN.linear_forward(x, self.w.weight(dtype), self.b.w if self.b else None, y, self.act, self.pre)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dtype = dtype_of(x)
        dx = new_like(dtype) if _needs(node, 0) else None
        myNN.linear_backward(x, self.w.weight(dtype), node.realize_cached_data(), out_grad.realize_cached_data(),
                             self.act, self.pre, dx, self.w.g, self.b.g if self.b else None)
        return (const(dx) if dx is not None else None,)

    def release(self):
        self.pre = None


class LayerNorm(NNOp):
    has_params = True

    def __init__(self, gamma, beta, eps=1e-6):
        self.gamma, self.beta, self.eps = gamma, beta, eps

    def compute(self, x):
        y = new_like(dtype_of(x))
        self.mean_rstd = tf([1], "gpu")
        myNN.layernorm_forward(x, self.gamma.w, self.beta.w, y, self.mean_rstd, self.eps)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.layernorm_backward(x, out_grad.realize_cached_data(), self.gamma.w, self.mean_rstd, dx, self.gamma.g,
                                self.beta.g)
        return (const(dx),)

    def release(self):
        self.mean_rstd = None


class Attention(NNOp):
    """输入 qkv [B, L, 3, H, hd]，输出 [B, L, H, hd]；softmax(QK^T / sqrt(hd)) V，整个 (b, h) 在一个 block 的 smem 内"""

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
        myNN.attention_backward(qkv, self.probs, out_grad.realize_cached_data(), self.heads, dqkv)
        return (const(dqkv),)

    def release(self):
        self.probs = None


class Tokens(NNOp):
    """[B, N, D] 的 patch 特征前面拼上 cls token，再加位置编码 -> [B, N+1, D]"""
    has_params = True

    def __init__(self, cls, pos):
        self.cls, self.pos = cls, pos

    def compute(self, patches):
        dtype, out = dtype_of(patches), new_like(dtype_of(patches))
        myNN.tokens_forward(patches, self.cls.weight(dtype), self.pos.weight(dtype), out)
        return out

    def gradient(self, out_grad, node):
        dp = new_like(dtype_of(out_grad.realize_cached_data()))
        myNN.tokens_backward(out_grad.realize_cached_data(), dp, self.cls.g, self.pos.g)
        return (const(dp),)


class SelectToken(NNOp):
    """取 cls token：[B, L, D] -> [B, D]"""

    def compute(self, x):
        y = new_like(dtype_of(x))
        myNN.select_token_forward(x, y)
        return y

    def gradient(self, out_grad, node):
        x = node.inputs[0].realize_cached_data()
        dx = new_like(dtype_of(x))
        myNN.select_token_backward(out_grad.realize_cached_data(), dx, x.get_shape())
        return (const(dx),)


class Add(NNOp):
    def compute(self, a, b):
        out = new_like(dtype_of(a))
        myNN.add(a, b, out)
        return out

    def gradient(self, out_grad, node):
        return out_grad, out_grad


class View(NNOp):
    """不拷贝的 reshape：输出与输入共享显存（view 会让输入的 Python 对象保持存活）"""

    def __init__(self, shape):
        self.shape = list(shape)

    def compute(self, x):
        self.in_shape = x.get_shape()
        return x.view(0, self.shape)

    def gradient(self, out_grad, node):
        return (const(out_grad.realize_cached_data().view(0, self.in_shape)),)
