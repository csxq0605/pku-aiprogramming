"""
Tensor：自动微分的计算图节点，以及与布局无关的基础算子（加减乘除、求和、矩阵乘、变形等，来自 HW4）
网络层用到的算子在 tinytensor.nn.functional
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from . import _state
from .device import cpu, Device
from .autograd.function import Op, Value
from .autograd.engine import compute_gradient_of_variables
from ._C import myTensor, myNN
from ._C.myTensor import Tensor_float as tf
from ._C.myTensor import Tensor_int as ti
from ._C.myNN import Tensor_bf16 as tb, Tensor_uint8 as tu

GPU_TYPES = (tf, ti, tb, tu)

class Tensor(Value):
    grad: "Tensor"
    # True：数据是 NHWC 布局，后续算子走 tensor core 那一组 kernel（见 nn.functional）；由算子向输出传递
    channels_last = False

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

        if kwargs.get("memory_format") == _state.channels_last:
            self.channels_last = True
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
        if isinstance(data, tb):
            return "bfloat16"
        if isinstance(data, tu):
            return np.dtype(np.uint8)
        return data.dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None, retain_graph=False):
        """
        反向传播，梯度写进叶子节点的 grad（参数的梯度写进 Parameter）。
        与 PyTorch 一样默认释放计算图：中间节点的梯度一传给输入就释放，它的前向结果也随之释放；
        retain_graph=True 时保留所有中间结果和每个节点的 grad
        """
        if out_grad is None:
            array = np.ones(self.shape, dtype=np.float32)
            out_grad = Tensor(array, device=self.device, requires_grad=False)
        compute_gradient_of_variables(self, out_grad, free_graph=not retain_graph)


    def __repr__(self):
        return "Tensor(" + str(self.numpy()) + ")"

    def __str__(self):
        return self.numpy().__str__()

    def numpy(self):
        return _np(self.realize_cached_data())


    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.channels_last:
                from .nn import functional as F
                return F.add(self, other)
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

    def view(self, *shape):
        """不拷贝的变形，输出与输入共享显存"""
        from .nn import functional as F
        return F.view(self, shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape)

    def flatten(self, start_dim=1):
        from .nn import functional as F
        return F.flatten(self, start_dim)

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
        if isinstance(a, tb) and isinstance(b, tb):
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
            axes = tuple(a % len(input_shape) for a in axes)
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


def tensor(data, requires_grad=False, memory_format=None):
    """
    把 numpy 数组上传到 GPU（浮点 -> fp32，整数 -> int32），或直接包装一个已经在 GPU 上的张量。
    memory_format=channels_last 表示数据是 NHWC 布局
    """
    if isinstance(data, Tensor):
        data = data.realize_cached_data()
    if not isinstance(data, GPU_TYPES):
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            data = ti(np.ascontiguousarray(data, dtype=np.int32), "gpu")
        else:
            data = tf(np.ascontiguousarray(data, dtype=np.float32), "gpu")
    return Tensor(data, requires_grad=requires_grad, memory_format=memory_format)


class no_grad:
    """with no_grad(): 期间创建的节点不需要梯度（评估时用）"""

    def __enter__(self):
        self.prev = _state.grad_enabled
        _state.grad_enabled = False

    def __exit__(self, *exc):
        _state.grad_enabled = self.prev


def is_grad_enabled():
    return _state.grad_enabled
