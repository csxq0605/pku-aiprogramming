"""
本文件我们给出一个基本完善的Tensor类
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from autodiff import compute_gradient_of_variables
from myTensor import Tensor_float as tf
from myTensor import Tensor_int as ti
import myLayer as ml
try:
    # 图像模型模块（bf16 / uint8 张量）；没编译时不影响 MNIST 部分
    from myNN import Tensor_bf16 as tb, Tensor_uint8 as tu
    import myNN
except ImportError:
    tb = tu = myNN = None
GPU_TYPES = tuple(t for t in (tf, ti, tb, tu) if t is not None)

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        elif isinstance(array, np.ndarray):
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        elif isinstance(array, GPU_TYPES):
            # 直接持有 GPU 张量，不拷回 CPU
            device = device if device else cpu()
            cached_data = array

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return np.array(numpy_array, dtype=dtype)
    
    @staticmethod
    def _array_from_mytensor(mytensor, device, dtype):
        return np.asarray(mytensor.numpy(), dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    def realize_cached_data(self):
        """
        GPU 算子（op.gpu = True）直接接收 GPU 张量；
        其余 numpy 实现的算子先把输入拷回 numpy
        """
        if self.cached_data is not None:
            return self.cached_data
        inputs = [x.realize_cached_data() for x in self.inputs]
        if not getattr(self.op, "gpu", False):
            inputs = [_np(a) for a in inputs]
        self.cached_data = self.op.compute(*inputs)
        return self.cached_data

    @property
    def shape(self):
        data = self.realize_cached_data()
        if isinstance(data, GPU_TYPES):
            return tuple(data.get_shape())
        return data.shape

    @property
    def dtype(self):
        data = self.realize_cached_data()
        if isinstance(data, tf):
            return np.dtype(np.float32)
        if isinstance(data, ti):
            return np.dtype(np.int32)
        if tb is not None and isinstance(data, tb):
            return "bfloat16"
        if tu is not None and isinstance(data, tu):
            return np.dtype(np.uint8)
        return data.dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        if out_grad is None:
            array = np.ones(self.shape, dtype=np.float32)
            out_grad = Tensor(array, device=self.device, requires_grad=False)
        compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.numpy()) + ")"

    def __str__(self):
        return self.numpy().__str__()

    def numpy(self):
        return _np(self.realize_cached_data())


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


def _gpu(array):
    """转成 GPU 上的 Tensor_float；已经在 GPU 上则原样返回（注意不要原地修改返回值）"""
    if isinstance(array, tf):
        return array
    return tf(np.ascontiguousarray(array, dtype=np.float32), "gpu")


def _np(array):
    """转成 numpy 数组"""
    if isinstance(array, GPU_TYPES):
        return array.numpy()
    return array


def _grad(data):
    """把算子算出的梯度包装成不需要再求导的 Tensor，GPU 上的梯度留在 GPU"""
    if not isinstance(data, GPU_TYPES):
        data = np.asarray(data, dtype=np.float32)
    return Tensor(data, requires_grad=False)


class EWiseAdd(TensorOp):
    gpu = True

    def compute(self, a, b):
        if isinstance(a, tf) and isinstance(b, tf) and a.get_shape() == b.get_shape():
            return a + b
        if tb is not None and isinstance(a, tb) and isinstance(b, tb):
            out = tb([1], "gpu")
            myNN.add(a, b, out)
            return out
        return _np(a) + _np(b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: np.ndarray) -> np.ndarray:
        return np.power(a, self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        return a / b
        

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: Tensor):
        axes = self.axes
        if axes is None:
            axs = list(range(len(a.shape) - 2))
            axs.extend([len(a.shape) - 1, len(a.shape) - 2])
        else:
            x1, x2 = axes
            axs = list(range(len(a.shape)))
            axs[x1], axs[x2] = axs[x2], axs[x1]
        return np.transpose(a, axs)

    def gradient(self, out_grad, node):
        axes = self.axes
        return transpose(out_grad, axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    gpu = True

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if isinstance(a, tf):
            new_shape = list(np.empty(a.get_shape(), dtype=np.bool_).reshape(self.shape).shape)
            return tf(a).resize(new_shape)
        return np.reshape(a, self.shape)
        

    def gradient(self, out_grad, node):
        orginal_shape = node.inputs[0].shape
        return reshape(out_grad, orginal_shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return np.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        shape = [1] * (len(self.shape) - len(input_shape)) + list(input_shape)
        axes = []
        for i, s in enumerate(self.shape):
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        return (reshape(summation(out_grad, axes = tuple(axes)), input_shape),)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = self.axes
        if axes is None:
            return np.sum(a)
        else:
            return np.sum(a, axis=axes)
        

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(input_shape)))
        else:
            axes = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
        grad_shape = [1 if i in axes else input_shape[i] for i in range(len(input_shape))]
        return broadcast_to(reshape(out_grad, grad_shape), input_shape)
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return np.matmul(a, b)
        

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        if len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes = tuple(range(len(grad_a.shape) - len(a.shape))))
        if len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes = tuple(range(len(grad_b.shape) - len(b.shape))))
        return grad_a, grad_b
        

def matmul(a, b):
    return MatMul()(Tensor(a), Tensor(b))


class Negate(TensorOp):
    def compute(self, a):
        return -a
        

    def gradient(self, out_grad, node):
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return np.log(a)
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return np.exp(a)
        

    def gradient(self, out_grad, node):
        return out_grad * np.exp(node.inputs[0].realize_cached_data())
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    gpu = True

    def compute(self, a):
        return _gpu(a).ReluForward()

    def gradient(self, out_grad, node):
        input = _gpu(node.inputs[0].realize_cached_data())
        grad_output = _gpu(out_grad.realize_cached_data())
        return _grad(input.ReluBackward(grad_output))


def relu(a):
    return ReLU()(a)


class FC(TensorOp):
    gpu = True

    def __init__(self, use_bias):
        self.use_bias = use_bias

    def compute(self, input, weight, bias):
        input_tensor = _gpu(input)
        weight_tensor = _gpu(weight)
        bias_tensor = _gpu(bias)
        if not self.use_bias:
            bias_tensor = tf(bias_tensor).zeros()
        output_shape = [input_tensor.get_shape()[0], weight_tensor.get_shape()[1]]
        output = tf(output_shape, "gpu")
        ml.FcForward(input_tensor, output, weight_tensor, bias_tensor)
        return output
    
    def gradient(self, out_grad, node):
        input, weight, bias = node.inputs
        input_tensor = _gpu(input.realize_cached_data())
        weight_tensor = _gpu(weight.realize_cached_data())
        bias_tensor = _gpu(bias.realize_cached_data())
        grad_output = _gpu(out_grad.realize_cached_data())
        grad_weight = tf(weight_tensor.get_shape(), "gpu")
        grad_bias = tf(bias_tensor.get_shape(), "gpu")
        if not input.requires_grad:
            # 输入不需要梯度（例如网络的输入数据）：只计算参数梯度
            ml.FcBackwardParams(input_tensor, weight_tensor, bias_tensor, grad_output, grad_weight, grad_bias)
            if not self.use_bias:
                grad_bias.zeros()
            return None, _grad(grad_weight), _grad(grad_bias)
        grad_input = tf(input_tensor.get_shape(), "gpu")
        # 第二个参数（前向输出）在反向中用不到，传 grad_output 占位，避免多分配一块显存
        ml.FcBackward(input_tensor, grad_output, weight_tensor, bias_tensor, grad_input, grad_output, grad_weight, grad_bias)
        if not self.use_bias:
            grad_bias.zeros()
        return _grad(grad_input), _grad(grad_weight), _grad(grad_bias)



def fc(input, weight, bias, use_bias=True):
    return FC(use_bias)(input, weight, bias)


class FCReLU(TensorOp):
    """relu(fc(x))：bias 与 ReLU 在一个 kernel 里完成，反向时 ReLU 掩码与 bias 梯度也在一个 kernel 里完成"""
    gpu = True

    def compute(self, input, weight, bias):
        output = tf([1], "gpu")
        ml.FcReluForward(_gpu(input), output, _gpu(weight), _gpu(bias))
        return output

    def gradient(self, out_grad, node):
        input, weight, bias = node.inputs
        need_grad_input = input.requires_grad
        grad_input = tf([1], "gpu")
        grad_weight = tf([1], "gpu")
        grad_bias = tf([1], "gpu")
        ml.FcReluBackward(_gpu(input.realize_cached_data()), _gpu(node.realize_cached_data()),
                          _gpu(weight.realize_cached_data()), grad_input, _gpu(out_grad.realize_cached_data()),
                          grad_weight, grad_bias, need_grad_input)
        return (_grad(grad_input) if need_grad_input else None), _grad(grad_weight), _grad(grad_bias)


def fc_relu(input, weight, bias):
    return FCReLU()(input, weight, bias)


class Conv(TensorOp):
    gpu = True

    def __init__(self, pad_h, pad_w, stride_h, stride_w):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w

    def compute(self, input, weight):
        input_tensor = _gpu(input)
        weight_tensor = _gpu(weight)
        output_shape = [input_tensor.get_shape()[0], weight_tensor.get_shape()[0],
                        (input_tensor.get_shape()[2] + 2 * self.pad_h - weight_tensor.get_shape()[2]) // self.stride_h + 1,
                        (input_tensor.get_shape()[3] + 2 * self.pad_w - weight_tensor.get_shape()[3]) // self.stride_w + 1]
        output = tf(output_shape, "gpu")
        ml.ConvForward(input_tensor, output, weight_tensor, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return output
    
    def gradient(self, out_grad, node):
        input, weight = node.inputs
        input_tensor = _gpu(input.realize_cached_data())
        weight_tensor = _gpu(weight.realize_cached_data())
        grad_output = _gpu(out_grad.realize_cached_data())
        grad_weight = tf(weight_tensor.get_shape(), "gpu")
        if not input.requires_grad:
            # 输入不需要梯度（例如网络的输入图像）：只计算卷积核梯度
            ml.ConvBackwardWeight(input_tensor, weight_tensor, grad_output, grad_weight, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
            return None, _grad(grad_weight)
        grad_input = tf(input_tensor.get_shape(), "gpu")
        ml.ConvBackward(input_tensor, grad_output, weight_tensor, grad_input, grad_output, grad_weight, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return _grad(grad_input), _grad(grad_weight)



def conv(input, weight, pad_h = 0, pad_w = 0, stride_h = 1, stride_w = 1):
    return Conv(pad_h, pad_w, stride_h, stride_w)(input, weight)


class MaxPool(TensorOp):
    gpu = True

    def __init__(self, kernel_shape, pad_h, pad_w, stride_h, stride_w):
        self.kernel_shape = kernel_shape
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.mask = None

    def compute(self, input):
        input_tensor = _gpu(input)
        output_shape = [input_tensor.get_shape()[0], input_tensor.get_shape()[1],
                        (input_tensor.get_shape()[2] + 2 * self.pad_h - self.kernel_shape[0]) // self.stride_h + 1,
                        (input_tensor.get_shape()[3] + 2 * self.pad_w - self.kernel_shape[1]) // self.stride_w + 1]
        output = tf(output_shape, "gpu")
        self.mask = tf(output_shape, "gpu")
        ml.MaxPoolingForward(input_tensor, output, self.mask, self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return output
    
    def gradient(self, out_grad, node):
        input = node.inputs[0]
        input_shape = list(input.shape)
        grad_output = _gpu(out_grad.realize_cached_data())
        grad_input = tf(input_shape, "gpu")
        ml.MaxPoolingBackward(grad_output, self.mask, grad_input, input_shape, self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return _grad(grad_input)
    


def maxpool(input, kernel_shape = [2, 2], pad_h = 0, pad_w = 0, stride_h = 2, stride_w = 2):
    return MaxPool(kernel_shape, pad_h, pad_w, stride_h, stride_w)(input)


class ConvReluMaxPool(TensorOp):
    """
    maxpool(relu(conv(x)))，池化窗口 2x2、步长 2。三步在一个 kernel 里完成：卷积结果只留在寄存器里，不写回显存；
    反向由 argmax 直接还原卷积输出的梯度，不需要清零和 atomicAdd
    """
    gpu = True

    def __init__(self, pad_h, pad_w, stride_h, stride_w):
        self.pad_h, self.pad_w, self.stride_h, self.stride_w = pad_h, pad_w, stride_h, stride_w
        self.argmax = None

    def compute(self, input, weight):
        output = tf([1], "gpu")
        self.argmax = ti([1], "gpu")
        ml.ConvReluMaxPoolForward(_gpu(input), _gpu(weight), output, self.argmax,
                                  self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return output

    def gradient(self, out_grad, node):
        input, weight = node.inputs
        need_grad_input = input.requires_grad
        grad_input = tf([1], "gpu")
        grad_weight = tf([1], "gpu")
        ml.ConvReluMaxPoolBackward(_gpu(input.realize_cached_data()), _gpu(weight.realize_cached_data()),
                                   _gpu(out_grad.realize_cached_data()), self.argmax, grad_input, grad_weight,
                                   self.pad_h, self.pad_w, self.stride_h, self.stride_w, need_grad_input)
        return (_grad(grad_input) if need_grad_input else None), _grad(grad_weight)


def conv_relu_maxpool(input, weight, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    """等价于 maxpool(relu(conv(...)))；通道数较大（Cin * kh * kw > 256）时退回三个独立算子"""
    _, c_in, kh, kw = weight.shape
    if c_in * kh * kw > 256:
        return maxpool(relu(conv(input, weight, pad_h, pad_w, stride_h, stride_w)))
    return ConvReluMaxPool(pad_h, pad_w, stride_h, stride_w)(input, weight)


class Softmaxloss(TensorOp):
    """numpy 实现的 softmax 交叉熵，返回 batch 平均 loss（作为参考实现）"""

    def __init__(self, labels):
        self.softmax = None
        self.labels = _np(labels.realize_cached_data()).astype(np.int64)
        self.batchsize = self.labels.shape[0]

    def compute(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        log_prob = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
        self.softmax = np.exp(log_prob)
        loss = -np.mean(log_prob[np.arange(self.batchsize), self.labels])
        return np.array(loss, dtype=np.float32)

    def gradient(self, out_grad, node):
        y_onehot = np.zeros_like(self.softmax)
        y_onehot[np.arange(self.batchsize), self.labels] = 1
        dZ = (self.softmax - y_onehot) / self.batchsize
        return _grad(dZ * out_grad.numpy())


def softmaxloss(X, y):
    return Softmaxloss(y)(X)


class Softmaxcrossentropyloss(TensorOp):
    """CUDA 实现的 softmax 交叉熵，返回 GPU 上的 batch 平均 loss（形状 [1]），读取时才同步"""
    gpu = True

    def __init__(self, labels):
        labels = labels.realize_cached_data()
        if isinstance(labels, ti):
            self.labels = labels
        else:
            self.labels = ti(np.ascontiguousarray(labels, dtype=np.int32), "gpu")
        self.grad = None

    def compute(self, input):
        # 一个 kernel 同时算出 batch 平均 loss 和对 logits 的梯度 (softmax - onehot) / N
        loss = tf([1], "gpu")
        self.grad = tf([1], "gpu")
        ml.SoftmaxCrossEntropy(_gpu(input), self.labels, loss, self.grad)
        return loss

    def gradient(self, out_grad, node):
        scale = float(np.asarray(_np(out_grad.realize_cached_data())).reshape(-1)[0])
        if scale == 1.0:
            return _grad(self.grad)
        return _grad(tf(self.grad).mults(scale))


def softmaxcrossentropyloss(X, y):
    return Softmaxcrossentropyloss(y)(X)
