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
        elif isinstance(array, tf) or isinstance(array, ti):
            device = device if device else cpu()
            #print("Tensor from myTensor")
            cached_data = Tensor._array_from_mytensor(array, device=device, dtype=dtype)

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
        return np.array(mytensor.get_data(), dtype=dtype).reshape(mytensor.get_shape())

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

    @property
    def shape(self):
        if isinstance(self.cached_data, tuple):
            return [1]
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        if isinstance(self.cached_data, tuple):
            return float
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        if out_grad is None:
            array = np.ones(self.shape, dtype=self.dtype)
            out_grad = Tensor(array, device=self.device, dtype=self.dtype, requires_grad=False)
        compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        return data


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


class EWiseAdd(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

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
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
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
    def compute(self, a):
        input = tf(a, "gpu")
        output = input.ReluForward()
        output = np.array(output.get_data()).reshape(input.get_shape())
        return output
        

    def gradient(self, out_grad, node):
        input = tf(node.inputs[0].realize_cached_data(), "gpu")
        grad_output = tf(out_grad.realize_cached_data(), "gpu")
        grad_input = input.ReluBackward(grad_output)
        return Tensor(grad_input)
        


def relu(a):
    return ReLU()(a)


class FC(TensorOp):
    def __init__(self, use_bias):
        self.use_bias = use_bias

    def compute(self, input, weight, bias):
        input_tensor = tf(input, "gpu")
        weight_tensor = tf(weight, "gpu")
        bias_tensor = tf(bias, "gpu")
        if not self.use_bias:
            bias_tensor.zeros()
        output_shape = [input_tensor.get_shape()[0], weight_tensor.get_shape()[1]]
        output = tf(output_shape, "gpu")
        ml.FcForward(input_tensor, output, weight_tensor, bias_tensor)
        output = np.array(output.get_data()).reshape(output_shape)
        return output
    
    def gradient(self, out_grad, node):
        input, weight, bias = node.inputs
        input_tensor = tf(input.realize_cached_data(), "gpu")
        weight_tensor = tf(weight.realize_cached_data(), "gpu")
        bias_tensor = tf(bias.realize_cached_data(), "gpu")
        grad_output = tf(out_grad.realize_cached_data(), "gpu")
        output = tf(grad_output.get_shape(), "gpu")
        grad_input = tf(input_tensor.get_shape(), "gpu")
        grad_weight = tf(weight_tensor.get_shape(), "gpu")
        grad_bias = tf(bias_tensor.get_shape(), "gpu")
        ml.FcBackward(input_tensor, output, weight_tensor, bias_tensor, grad_input, grad_output, grad_weight, grad_bias)
        if not self.use_bias:
            grad_bias.zeros()
        return Tensor(grad_input), Tensor(grad_weight), Tensor(grad_bias)



def fc(input, weight, bias, use_bias=True):
    return FC(use_bias)(input, weight, bias)


class Conv(TensorOp):
    def __init__(self, pad_h, pad_w, stride_h, stride_w):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w

    def compute(self, input, weight):
        input_tensor = tf(input, "gpu")
        weight_tensor = tf(weight, "gpu")
        output_shape = [input_tensor.get_shape()[0], weight_tensor.get_shape()[0],
                        (input_tensor.get_shape()[2] + 2 * self.pad_h - weight_tensor.get_shape()[2]) // self.stride_h + 1,
                        (input_tensor.get_shape()[3] + 2 * self.pad_w - weight_tensor.get_shape()[3]) // self.stride_w + 1]
        output = tf(output_shape, "gpu")
        ml.ConvForward(input_tensor, output, weight_tensor, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        output = np.array(output.get_data()).reshape(output_shape)
        return output
    
    def gradient(self, out_grad, node):
        input, weight = node.inputs
        input_tensor = tf(input.realize_cached_data(), "gpu")
        weight_tensor = tf(weight.realize_cached_data(), "gpu")
        grad_output = tf(out_grad.realize_cached_data(), "gpu")
        output = tf(grad_output.get_shape(), "gpu")
        grad_input = tf(input_tensor.get_shape(), "gpu")
        grad_weight = tf(weight_tensor.get_shape(), "gpu")
        ml.ConvBackward(input_tensor, output, weight_tensor, grad_input, grad_output, grad_weight, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return Tensor(grad_input), Tensor(grad_weight)



def conv(input, weight, pad_h = 0, pad_w = 0, stride_h = 1, stride_w = 1):
    return Conv(pad_h, pad_w, stride_h, stride_w)(input, weight)


class MaxPool(TensorOp):
    def __init__(self, kernel_shape, pad_h, pad_w, stride_h, stride_w):
        self.kernel_shape = kernel_shape
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.mask = None

    def compute(self, input):
        input_tensor = tf(input, "gpu")
        output_shape = [input_tensor.get_shape()[0], input_tensor.get_shape()[1],
                        (input_tensor.get_shape()[2] + 2 * self.pad_h - self.kernel_shape[0]) // self.stride_h + 1,
                        (input_tensor.get_shape()[3] + 2 * self.pad_w - self.kernel_shape[1]) // self.stride_w + 1]
        output = tf(output_shape, "gpu")
        self.mask = tf(output_shape, "gpu")
        ml.MaxPoolingForward(input_tensor, output, self.mask, self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        output = np.array(output.get_data()).reshape(output_shape)
        return output
    
    def gradient(self, out_grad, node):
        input = node.inputs[0]
        input_shape = input.realize_cached_data().shape
        grad_output = tf(out_grad.realize_cached_data(), "gpu")
        grad_input = tf(input_shape, "gpu")
        ml.MaxPoolingBackward(grad_output, self.mask, grad_input, input_shape, self.kernel_shape, self.pad_h, self.pad_w, self.stride_h, self.stride_w)
        return Tensor(grad_input)
    


def maxpool(input, kernel_shape = [2, 2], pad_h = 0, pad_w = 0, stride_h = 2, stride_w = 2):
    return MaxPool(kernel_shape, pad_h, pad_w, stride_h, stride_w)(input)


class Softmaxloss(TensorOp):
    def __init__(self, labels):
        self.softmax = None
        self.labels = labels.realize_cached_data()
        self.batchsize = self.labels.shape[0]

    def compute(self, z):
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        z = z_exp / np.sum(z_exp, axis=1, keepdims=True)
        self.softmax = z
        error = np.sum(z.argmax(axis=1) != self.labels) / (self.batchsize*10)
        loss = 0.0
        for i in range(self.batchsize):
            logits = z[i]
            logits = logits - np.max(logits)
            correct_logit = logits[self.labels[i]]
            loss += np.log(np.sum(np.exp(logits - correct_logit))) / (self.batchsize*10)
        return loss, error
    
    def gradient(self, out_grad, node):
        y_onehot = np.zeros_like(self.softmax)
        y_onehot[np.arange(self.batchsize), self.labels] = 1
        dZ = (self.softmax - y_onehot) / self.batchsize
        #print(dZ)
        return Tensor(dZ)
    


def softmaxloss(X, y):
    return Softmaxloss(y)(X)


class Softmaxcrossentropyloss(TensorOp):
    def __init__(self, labels):
        self.labels = labels.realize_cached_data()
        self.batchsize = self.labels.shape[0]
        self.softmax = None

    def compute(self, input):
        input_tensor = tf(input, "gpu")
        labels_tensor = ti(self.labels.shape, "gpu", self.labels)
        output = tf(input_tensor.get_shape(), "gpu")
        ml.SoftmaxForward(input_tensor, output)
        self.softmax = np.array(output.get_data()).reshape(input_tensor.get_shape())
        error = np.sum(self.softmax.argmax(axis=1) != self.labels)/(10*self.batchsize)
        loss = tf([1], "gpu")
        ml.CrossEntropyLoss(output, labels_tensor, loss)
        loss = loss.get_data()[0]/(10*self.batchsize)
        return loss, error
        
    def gradient(self, out_grad, node):
        output = tf(self.softmax, "gpu")
        labels_tensor = ti(self.labels.shape, "gpu", self.labels)
        grad_input = tf(output.get_shape(), "gpu")
        #print("@backward")
        ml.CrossEntropyLossBackward(output, labels_tensor, grad_input)
        grad_input.mults(1/self.batchsize)
        return Tensor(grad_input)
    

def softmaxcrossentropyloss(X, y):
    return Softmaxcrossentropyloss(y)(X)