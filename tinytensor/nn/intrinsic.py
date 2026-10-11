"""
融合层（对应 torch.nn.intrinsic）：数学上等于几个基本层的串联，但只用一个算子 / 一个 kernel 完成，
省掉中间结果的读写和 kernel 发射。每个融合层都可以换成注释里的基本层组合，结果一致（tests/ 里有对照）
"""
from . import functional as F
from .module import Module
from .modules import BatchNorm2d, Conv2d, Linear


class LinearReLU(Linear):
    """ReLU(Linear(x))：bias 与 ReLU 在同一个 kernel 里完成，反向时 ReLU 掩码与 bias 梯度也在同一个 kernel 里"""
    activation = "relu"


class LinearGELU(Linear):
    """GELU(Linear(x))（channels_last）"""
    activation = "gelu"


class ConvReLUMaxPool2d(Conv2d):
    """MaxPool2d(2)(ReLU(Conv2d(x)))（contiguous_format）：卷积结果只留在寄存器里，不写回显存"""

    def forward(self, x):
        return F.conv_relu_max_pool2d(x, self.weight, self.stride, self.padding)


class ConvBNAct2d(Module):
    """
    [ReLU](BatchNorm2d(Conv2d(x)) [+ residual])（channels_last）
    fuse=True ：BN、残差相加、ReLU 在同一个 kernel 里完成（前向一个、反向一个）
    fuse=False：BN / 相加 / ReLU 各自一个 kernel（消融实验用）
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, relu=True,
                 fuse=True):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.bn = BatchNorm2d(out_channels)
        self.relu, self.fuse = relu, fuse

    def forward(self, x, residual=None):
        y = self.conv(x)
        if self.fuse:
            return self.bn(y, relu=self.relu, residual=residual)
        y = self.bn(y)
        if residual is not None:
            y = F.add(y, residual)
        return F.relu(y) if self.relu else y
