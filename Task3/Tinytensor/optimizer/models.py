"""
Tiny ImageNet（64x64，200 类）上的 VGG / ResNet / ResNeXt / ViT，基于 nn_ops（myNN）
- 布局 NHWC；卷积权重 [Cout, KH, KW, Cin/groups]，全连接权重 [out, in]
- 所有参数放在一块扁平显存里（权重 / 梯度 / 动量 / bf16 副本各一块），优化器一个 kernel 更新全部参数，
  梯度清零也只是一次 memset
- 初始化由 TinyTensor 自己在 GPU 上生成，可导出给 PyTorch 对照实现加载
PyTorch 对照实现（torch_imagenet.py）按同样的参数名和顺序构建同样的网络
"""
import numpy as np

import myNN
from myTensor import Tensor_float as tf
import nn_ops as F
from nn_ops import Param

ALIGN = 64  # 每个参数的起始位置按 64 个元素（256 字节）对齐，方便以后的向量化 / cp.async 访存


class BN:
    def __init__(self, model, name, c, momentum=0.1, eps=1e-5):
        self.gamma = model.param(name + ".weight", [c], "ones")
        self.beta = model.param(name + ".bias", [c], "zeros")
        self.running_mean = tf([c], "gpu").zeros()
        self.running_var = tf(np.ones(c, np.float32), "gpu")
        self.name, self.momentum, self.eps = name, momentum, eps
        model.bns.append(self)


class ImageModel:
    """子类在 build() 里用 self.param / self.conv_bn 等声明参数，在 forward() 里搭计算图"""
    name = None
    num_classes = 200

    def __init__(self, dtype="fp32", seed=0):
        assert dtype in ("fp32", "bf16")
        self.dtype, self.seed = dtype, seed
        self.params, self.bns = [], []
        self.build()
        self._allocate()

    # ---------- 参数 ----------
    def param(self, name, shape, init, fan_in=None):
        p = Param(name, shape, init, fan_in)
        self.params.append(p)
        return p

    def _allocate(self):
        offset = 0
        for p in self.params:
            p.offset = offset
            offset += (int(np.prod(p.shape)) + ALIGN - 1) // ALIGN * ALIGN
        self.numel = offset
        self.flat_w = tf([offset], "gpu").zeros()
        self.flat_g = tf([offset], "gpu").zeros()
        self.flat_lowp = myNN.Tensor_bf16([offset], "gpu").zeros() if self.dtype == "bf16" else None
        for i, p in enumerate(self.params):
            p.w = self.flat_w.view(p.offset, list(p.shape))
            p.g = self.flat_g.view(p.offset, list(p.shape))
            p.lowp = self.flat_lowp.view(p.offset, list(p.shape)) if self.flat_lowp is not None else None
            self._init_param(p, self.seed * 100003 + i)
        self.sync_lowp()

    def _init_param(self, p, seed):
        n = int(np.prod(p.shape))
        if p.init == "zeros":
            return
        if p.init == "ones":
            p.w.copy_from(tf(np.ones(n, np.float32).reshape(p.shape), "gpu"), 0)
            return
        if p.init == "kaiming":       # 卷积：kaiming uniform（ReLU 增益），方差 2 / fan_in
            bound = np.sqrt(6.0 / p.fan_in)
        elif p.init == "fanin":       # 分类头：PyTorch nn.Linear 默认的 U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            bound = 1.0 / np.sqrt(p.fan_in)
        elif p.init == "xavier":      # ViT 的全连接：xavier uniform
            bound = np.sqrt(6.0 / (p.fan_in + p.shape[0]))
        elif p.init == "std02":       # ViT 的 cls / 位置编码：标准差 0.02 的均匀分布
            bound = 0.02 * np.sqrt(3.0)
        else:
            raise ValueError(p.init)
        p.w.random(-float(bound), float(bound), seed)

    def sync_lowp(self):
        """把 fp32 主权重整体转成 bf16 副本（之后由优化器 kernel 在更新时顺便写）"""
        if self.flat_lowp is not None:
            myNN.cast(self.flat_w, self.flat_lowp)

    def export_init(self, path):
        np.savez(path, **{p.name: p.w.numpy() for p in self.params})

    def load_params(self, path):
        data = np.load(path)
        for p in self.params:
            p.w.copy_from(tf(np.ascontiguousarray(data[p.name], np.float32), "gpu"), 0)
        self.sync_lowp()

    # ---------- 常用积木 ----------
    def conv_bn(self, name, cin, cout, k, stride=1, groups=1, pad=None):
        w = self.param(name + ".conv.weight", [cout, k, k, cin // groups], "kaiming", fan_in=k * k * cin // groups)
        bn = BN(self, name + ".bn", cout)
        return dict(w=w, bn=bn, stride=stride, pad=(k - 1) // 2 if pad is None else pad, groups=groups)

    def apply_conv_bn(self, x, layer, relu=True, residual=None):
        y = F.Conv(layer["w"], layer["stride"], layer["pad"], layer["groups"])(x)
        op = F.BatchNorm(layer["bn"], relu, self.training)
        return op(y, residual) if residual is not None else op(y)

    def linear(self, name, fin, fout, init="fanin", bias=True):
        w = self.param(name + ".weight", [fout, fin], init, fan_in=fin)
        b = self.param(name + ".bias", [fout], "zeros") if bias else None
        return w, b

    # ---------- 前向 ----------
    def __call__(self, x, training):
        self.training = training
        return self.forward(x)

    def build(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class VGG16(ImageModel):
    """VGG-16-BN；64x64 输入经过 5 次 2x2 池化到 2x2，全局平均池化后接一个全连接分类层"""
    name = "vgg16"
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def build(self):
        self.layers, cin = [], 3
        for i, v in enumerate(self.cfg):
            if v == "M":
                self.layers.append("M")
            else:
                self.layers.append(self.conv_bn("features.{}".format(i), cin, v, 3))
                cin = v
        self.fc = self.linear("fc", cin, self.num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.MaxPool2x2()(x) if layer == "M" else self.apply_conv_bn(x, layer)
        x = F.GlobalAvgPool()(x)
        return F.Linear(*self.fc, out_f32=True)(x)


class ResNet18(ImageModel):
    """ResNet-18：stem = 3x3 卷积（步长 1）+ 2x2 最大池化（64 -> 32），四个 stage 后 4x4，全局平均池化 + 全连接"""
    name = "resnet18"
    widths, blocks = (64, 128, 256, 512), (2, 2, 2, 2)

    def build(self):
        self.stem = self.conv_bn("stem", 3, 64, 3)
        self.stages, cin = [], 64
        for s, (width, n) in enumerate(zip(self.widths, self.blocks)):
            for b in range(n):
                stride = 2 if s > 0 and b == 0 else 1
                self.stages.append(self.block("layer{}.{}".format(s + 1, b), cin, width, stride))
                cin = self.out_channels(width)
        self.fc = self.linear("fc", cin, self.num_classes)

    def out_channels(self, width):
        return width

    def block(self, name, cin, width, stride):
        blk = dict(c1=self.conv_bn(name + ".c1", cin, width, 3, stride),
                   c2=self.conv_bn(name + ".c2", width, width, 3))
        if stride != 1 or cin != width:
            blk["down"] = self.conv_bn(name + ".down", cin, width, 1, stride)
        return blk

    def run_block(self, x, blk):
        shortcut = self.apply_conv_bn(x, blk["down"], relu=False) if "down" in blk else x
        y = self.apply_conv_bn(x, blk["c1"])
        # 第二个 BN 融合了残差相加和 ReLU
        return self.apply_conv_bn(y, blk["c2"], relu=True, residual=shortcut)

    def forward(self, x):
        x = F.MaxPool2x2()(self.apply_conv_bn(x, self.stem))
        for blk in self.stages:
            x = self.run_block(x, blk)
        x = F.GlobalAvgPool()(x)
        return F.Linear(*self.fc, out_f32=True)(x)


class ResNeXt26(ResNet18):
    """ResNeXt-26 (32x4d)：瓶颈块 [2, 2, 2, 2]，中间的 3x3 是 32 组的分组卷积；stem 与 ResNet18 相同"""
    name = "resnext26"
    cardinality, base_width = 32, 4

    def out_channels(self, width):
        return width * 4

    def block(self, name, cin, width, stride):
        mid = width * self.base_width // 64 * self.cardinality
        cout = self.out_channels(width)
        blk = dict(c1=self.conv_bn(name + ".c1", cin, mid, 1),
                   c2=self.conv_bn(name + ".c2", mid, mid, 3, stride, groups=self.cardinality),
                   c3=self.conv_bn(name + ".c3", mid, cout, 1))
        if stride != 1 or cin != cout:
            blk["down"] = self.conv_bn(name + ".down", cin, cout, 1, stride)
        return blk

    def run_block(self, x, blk):
        shortcut = self.apply_conv_bn(x, blk["down"], relu=False) if "down" in blk else x
        y = self.apply_conv_bn(x, blk["c1"])
        y = self.apply_conv_bn(y, blk["c2"])
        return self.apply_conv_bn(y, blk["c3"], relu=True, residual=shortcut)


class ViTTiny(ImageModel):
    """ViT-Tiny：8x8 patch -> 64 个 token + cls，D=192，3 头，12 层 pre-norm block，MLP 比例 4"""
    name = "vit_tiny"
    patch, dim, heads, depth, mlp_ratio = 8, 192, 3, 12, 4

    def build(self):
        D, P = self.dim, self.patch
        self.n_patches = (64 // P) ** 2
        self.patch_w = self.param("patch.weight", [D, P, P, 3], "xavier", fan_in=P * P * 3)
        self.cls = self.param("cls", [D], "std02")
        self.pos = self.param("pos", [self.n_patches + 1, D], "std02")
        self.blocks = []
        for i in range(self.depth):
            n = "blocks.{}.".format(i)
            self.blocks.append(dict(
                ln1=(self.param(n + "ln1.weight", [D], "ones"), self.param(n + "ln1.bias", [D], "zeros")),
                qkv=self.linear(n + "qkv", D, 3 * D, "xavier"),
                proj=self.linear(n + "proj", D, D, "xavier"),
                ln2=(self.param(n + "ln2.weight", [D], "ones"), self.param(n + "ln2.bias", [D], "zeros")),
                fc1=self.linear(n + "fc1", D, D * self.mlp_ratio, "xavier"),
                fc2=self.linear(n + "fc2", D * self.mlp_ratio, D, "xavier")))
        self.norm = (self.param("norm.weight", [D], "ones"), self.param("norm.bias", [D], "zeros"))
        self.head = self.linear("head", D, self.num_classes, "xavier")

    def forward(self, x):
        B, D, H = x.shape[0], self.dim, self.heads
        L = self.n_patches + 1
        x = F.Conv(self.patch_w, stride=self.patch)(x)            # [B, 8, 8, D]
        x = F.Tokens(self.cls, self.pos)(F.View([B, self.n_patches, D])(x))
        for blk in self.blocks:
            y = F.Linear(*blk["qkv"])(F.LayerNorm(*blk["ln1"])(x))
            y = F.Attention(H)(F.View([B, L, 3, H, D // H])(y))
            y = F.Linear(*blk["proj"])(F.View([B, L, D])(y))
            x = F.Add()(x, y)
            y = F.Linear(*blk["fc1"], act=2)(F.LayerNorm(*blk["ln2"])(x))
            x = F.Add()(x, F.Linear(*blk["fc2"])(y))
        x = F.SelectToken()(F.LayerNorm(*self.norm)(x))
        return F.Linear(*self.head, out_f32=True)(x)


MODELS = {m.name: m for m in (VGG16, ResNet18, ResNeXt26, ViTTiny)}
