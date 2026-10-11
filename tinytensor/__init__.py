"""
TinyTensor：一个小型的 CUDA 深度学习框架，组织方式与 PyTorch 相同

    import tinytensor as tt
    from tinytensor import nn, optim
    import tinytensor.nn.functional as F

  tt.Tensor / tt.tensor      计算图节点；基础算子（加减乘除、求和、矩阵乘等）
  tt.nn                      Module、Parameter 与各种层；tt.nn.intrinsic 是融合层
  tt.nn.functional           无状态的算子（每个对应一到两个 CUDA kernel）
  tt.optim                   SGD / Adam / AdamW 与学习率计划
  tt.utils.data              常驻 GPU 的数据加载
  tt.cuda                    同步、显存、CUDA Graph
  tt.models                  MLP / CNN（MNIST）与 VGG16 / ResNet18 / ResNeXt26 / ViT-Tiny
"""
from . import _C  # noqa: F401  先加载扩展模块
from . import _state
from ._state import bfloat16, channels_last, contiguous_format, float32  # noqa: F401
from .tensor import (Tensor, TensorOp, add, add_scalar, broadcast_to, divide, divide_scalar, exp,  # noqa: F401
                     is_grad_enabled, log, matmul, mul_scalar, multiply, negate, no_grad, power, power_scalar,
                     reshape, summation, tensor, transpose)
from . import autograd, cuda, nn, optim, utils  # noqa: F401
from . import models  # noqa: F401


def manual_seed(seed):
    """之后构造的模型用这个种子初始化参数（模型自己的 manual_seed() 优先）"""
    _state.seed = seed
