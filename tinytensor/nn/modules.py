"""
层（用法对应 torch.nn）：持有参数与状态，forward 调用 nn.functional 里的算子
每个层在两种内存布局下都只声明形状，实际的 kernel 由 functional 按布局选择
"""
import numpy as np

from ..tensor import tf, ti
from . import functional as F
from . import init
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """
    y = x W^T + b。权重的物理形状：contiguous_format 是 [in, out]，channels_last 是 [out, in]
    默认初始化与 torch.nn.Linear 的权重相同（U(-1 / sqrt(in), 1 / sqrt(in))），bias 为 0
    out_dtype=float32：混合精度下这一层输出 fp32（分类头的 logits）
    """
    activation = None

    def __init__(self, in_features, out_features, bias=True, out_dtype=None):
        super().__init__()
        self.in_features, self.out_features, self.out_dtype = in_features, out_features, out_dtype
        self.weight = Parameter((in_features, out_features), shape_cl=(out_features, in_features),
                                fan_in=in_features, fan_out=out_features)
        init.fan_in_uniform_(self.weight)
        self.bias = Parameter((out_features,)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias, self.activation, self.out_dtype)


class Conv2d(Module):
    """
    卷积（没有 bias：后面接 BatchNorm 或直接接激活）。Kaiming 均匀初始化
    权重的物理形状：contiguous_format 是 [Cout, Cin, K, K]，channels_last 是 [Cout, K, K, Cin / groups]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if bias:
            raise NotImplementedError("Conv2d has no bias")
        k, cin = kernel_size, in_channels // groups
        self.stride, self.padding, self.groups = stride, padding, groups
        self.weight = Parameter((out_channels, cin, k, k), shape_cl=(out_channels, k, k, cin),
                                fan_in=k * k * cin, fan_out=k * k * out_channels // groups)
        init.kaiming_uniform_(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding, self.groups)


class BatchNorm2d(Module):
    """channels_last 的 BatchNorm；参数与统计量始终是 fp32。relu / residual 见 F.batch_norm"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Parameter((num_features,), init="ones")
        self.bias = Parameter((num_features,))
        self.register_buffer("running_mean", tf([num_features], "gpu").zeros())
        self.register_buffer("running_var", tf(np.ones(num_features, np.float32), "gpu"))

    def forward(self, x, relu=False, residual=None):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training,
                            self.momentum, self.eps, relu, residual)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter((normalized_shape,), init="ones")
        self.bias = Parameter((normalized_shape,))

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class GlobalAvgPool2d(Module):
    def forward(self, x):
        return F.global_avg_pool2d(x)


class Flatten(Module):
    """从 start_dim 起展平；不拷贝，输出与输入共享显存"""

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return F.flatten(x, self.start_dim)


class CrossEntropyLoss(Module):
    """
    softmax 交叉熵。contiguous_format 下返回 batch 平均 loss；
    channels_last 下 loss 总和与预测正确数在 GPU 上累加（训练过程中不读回任何数值），
    用 statistics() 读取并清零 —— 通常每个 epoch 读一次
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.accumulators = None

    def forward(self, input, target):
        if not input.channels_last:
            return F.cross_entropy(input, target, self.label_smoothing)
        if self.accumulators is None:
            # 对 logits 的梯度、loss 总和、预测正确数
            self.accumulators = (tf([1], "gpu"), tf([1], "gpu").zeros(), ti([1], "gpu").zeros())
        return F.cross_entropy(input, target, self.label_smoothing, self.accumulators)

    def statistics(self):
        """(loss 总和, 预测正确数)，读取后清零（会同步 GPU）"""
        if self.accumulators is None:
            return 0.0, 0
        _, loss_sum, correct = self.accumulators
        stats = float(loss_sum.numpy()[0]), int(correct.numpy()[0])
        loss_sum.zeros()
        correct.zeros()
        return stats
