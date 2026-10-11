from .. import _state, nn
from ..nn import functional as F
from .vgg import ImageModel


class BasicBlock(nn.Module):
    """两个 3x3 卷积；第二个 BN 融合了残差相加和 ReLU"""
    expansion = 1

    def __init__(self, model, cin, width, stride):
        super().__init__()
        cout = width * self.expansion
        self.c1 = model.conv_bn(cin, width, 3, stride)
        self.c2 = model.conv_bn(width, width, 3)
        self.down = model.conv_bn(cin, cout, 1, stride, relu=False) if stride != 1 or cin != cout else None

    def forward(self, x):
        shortcut = self.down(x) if self.down is not None else x
        return self.c2(self.c1(x), residual=shortcut)


class Bottleneck(nn.Module):
    """ResNeXt 的瓶颈块：1x1 -> 3x3 分组卷积 -> 1x1，最后一个 BN 融合残差相加和 ReLU"""
    expansion, cardinality, base_width = 4, 32, 4

    def __init__(self, model, cin, width, stride):
        super().__init__()
        mid, cout = width * self.base_width // 64 * self.cardinality, width * self.expansion
        self.c1 = model.conv_bn(cin, mid, 1)
        self.c2 = model.conv_bn(mid, mid, 3, stride, groups=self.cardinality)
        self.c3 = model.conv_bn(mid, cout, 1)
        self.down = model.conv_bn(cin, cout, 1, stride, relu=False) if stride != 1 or cin != cout else None

    def forward(self, x):
        shortcut = self.down(x) if self.down is not None else x
        return self.c3(self.c2(self.c1(x)), residual=shortcut)


class ResNet18(ImageModel):
    """ResNet-18：stem = 3x3 卷积（步长 1）+ 2x2 最大池化（64 -> 32），四个 stage 后 4x4，全局平均池化 + 全连接"""
    name = "resnet18"
    block, widths, depths = BasicBlock, (64, 128, 256, 512), (2, 2, 2, 2)

    def build(self):
        self.stem, cin = self.conv_bn(3, 64, 3), 64
        self.stages = []
        for s, (width, depth) in enumerate(zip(self.widths, self.depths)):
            stage = nn.ModuleList()
            for b in range(depth):
                stage.append(self.block(self, cin, width, 2 if s > 0 and b == 0 else 1))
                cin = width * self.block.expansion
            setattr(self, "layer{}".format(s + 1), stage)
            self.stages.append(stage)
        self.fc = nn.Linear(cin, self.num_classes, out_dtype=_state.float32)

    def forward(self, x):
        x = F.max_pool2d(self.stem(x), 2)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.fc(F.global_avg_pool2d(x))


class ResNeXt26(ResNet18):
    """ResNeXt-26 (32x4d)：瓶颈块 [2, 2, 2, 2]，中间的 3x3 是 32 组的分组卷积；stem 与 ResNet18 相同"""
    name = "resnext26"
    block = Bottleneck
