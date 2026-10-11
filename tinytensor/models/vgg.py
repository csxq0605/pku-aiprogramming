from .. import _state, nn
from ..nn import intrinsic as nni


class ImageModel(nn.Module):
    """
    图像模型的公共部分：channels_last（NHWC，tensor core kernel）。
    fuse=False 时 BN / 残差相加 / ReLU、Linear / GELU 拆成独立算子（数学上相同，用来测融合省了多少）
    参数的名字和顺序与 PyTorch 对照实现（experiments/imagenet/torch_imagenet.py）一一对应，
    所以 TinyTensor 生成的初始权重可以直接导出给 PyTorch 加载
    """
    name = None
    seed_stride = 100003

    def __init__(self, num_classes=200, fuse=True):
        super().__init__()
        self.num_classes, self.fuse = num_classes, fuse
        self.build()
        self.to(memory_format=_state.channels_last)

    def build(self):
        raise NotImplementedError()

    def conv_bn(self, cin, cout, k, stride=1, groups=1, relu=True):
        return nni.ConvBNAct2d(cin, cout, k, stride, groups=groups, relu=relu, fuse=self.fuse)


class VGG16(ImageModel):
    """VGG-16-BN；64x64 输入经过 5 次 2x2 池化到 2x2，全局平均池化后接一个全连接分类层"""
    name = "vgg16"
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def build(self):
        self.features, cin = nn.Sequential(), 3
        for v in self.cfg:
            if v == "M":
                self.features.append(nn.MaxPool2d(2))
            else:
                self.features.append(self.conv_bn(cin, v, 3))
                cin = v
        self.pool = nn.GlobalAvgPool2d()
        self.fc = nn.Linear(cin, self.num_classes, out_dtype=_state.float32)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))
