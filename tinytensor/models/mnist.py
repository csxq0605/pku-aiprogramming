"""
MNIST 上的 MLP 与 CNN（contiguous_format，fp32）
fused=True 用融合层（FC + ReLU、卷积 + ReLU + 池化），数学上与 fused=False 的逐层写法完全相同
"""
from .. import nn
from ..nn import functional as F
from ..nn import intrinsic as nni


class MLP(nn.Module):
    """784 - 100 - 10"""

    def __init__(self, in_features=784, hidden=100, num_classes=10, fused=True):
        super().__init__()
        self.fused = fused
        self.flatten = nn.Flatten()
        self.fc1 = (nni.LinearReLU if fused else nn.Linear)(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        x = self.fc1(self.flatten(x))
        if not self.fused:
            x = F.relu(x)
        return self.fc2(x)


class CNN(nn.Module):
    """conv3x3(16) - pool - conv3x3(32) - pool - fc(128) - fc(10)，28x28 输入"""

    def __init__(self, num_classes=10, fused=True):
        super().__init__()
        self.fused = fused
        conv = nni.ConvReLUMaxPool2d if fused else nn.Conv2d
        self.conv1 = conv(1, 16, 3, padding=1)
        self.conv2 = conv(16, 32, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = (nni.LinearReLU if fused else nn.Linear)(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if self.fused:
            x = self.fc1(self.flatten(self.conv2(self.conv1(x))))
        else:
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)
