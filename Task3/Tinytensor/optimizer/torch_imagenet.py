"""
PyTorch 对照实验：与 imagenet_train.py（TinyTensor）相同的网络、初始权重、数据顺序、数据增强、学习率与优化器
- 初始权重读取 TinyTensor 导出的 results/imagenet/init/<arch>_seed<seed>.npz（先运行 TinyTensor）
- 训练集 uint8 常驻 GPU；随机裁剪 / 翻转用与 TinyTensor kernel 完全相同的整数哈希，所以每个 step 的 batch 逐像素一致
- fp32：关闭 TF32（与 TinyTensor 的 fp32 一样是真正的 fp32 运算）；bf16：torch.autocast + channels_last
- 运行方式：eager / --graph（torch.cuda.CUDAGraph 录连续 --chunk 个 step）/ --compile MODE（torch.compile 整个 step）
用法: python torch_imagenet.py --arch resnet18 --dtype bf16 [--graph | --compile default] [--bench 100]
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import imagenet_common as C

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("highest")


# ---------------- 网络（state_dict 的名字与 models.py 的参数名一一对应） ----------------
class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1, pad=None):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2 if pad is None else pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)

    def forward(self, x, relu=True, residual=None):
        y = self.bn(self.conv(x))
        if residual is not None:
            y = y + residual
        return F.relu(y) if relu else y


class VGG16(nn.Module):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self, num_classes=200):
        super().__init__()
        layers, cin = [], 3
        for v in self.cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(ConvBN(cin, v, 3))
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, cin, width, stride):
        super().__init__()
        self.c1, self.c2 = ConvBN(cin, width, 3, stride), ConvBN(width, width, 3)
        self.down = ConvBN(cin, width, 1, stride) if stride != 1 or cin != width else None

    def forward(self, x):
        shortcut = self.down(x, relu=False) if self.down is not None else x
        return self.c2(self.c1(x), relu=True, residual=shortcut)


class Bottleneck(nn.Module):
    def __init__(self, cin, width, stride, cardinality=32, base_width=4):
        super().__init__()
        mid, cout = width * base_width // 64 * cardinality, width * 4
        self.c1 = ConvBN(cin, mid, 1)
        self.c2 = ConvBN(mid, mid, 3, stride, groups=cardinality)
        self.c3 = ConvBN(mid, cout, 1)
        self.down = ConvBN(cin, cout, 1, stride) if stride != 1 or cin != cout else None

    def forward(self, x):
        shortcut = self.down(x, relu=False) if self.down is not None else x
        return self.c3(self.c2(self.c1(x)), relu=True, residual=shortcut)


class ResNet(nn.Module):
    def __init__(self, block, expansion, num_classes=200):
        super().__init__()
        self.stem = ConvBN(3, 64, 3)
        cin = 64
        for s, width in enumerate((64, 128, 256, 512)):
            blocks = []
            for b in range(2):
                blocks.append(block(cin, width, 2 if s > 0 and b == 0 else 1))
                cin = width * expansion
            setattr(self, "layer{}".format(s + 1), nn.Sequential(*blocks))
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.stem(x), 2)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(x.mean((2, 3)))


class ViTBlock(nn.Module):
    def __init__(self, D, H, ratio):
        super().__init__()
        self.H = H
        self.ln1, self.ln2 = nn.LayerNorm(D, eps=1e-6), nn.LayerNorm(D, eps=1e-6)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, D * ratio), nn.Linear(D * ratio, D)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, L, 3, self.H, D // self.H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])          # [B, H, L, hd]
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class ViTTiny(nn.Module):
    def __init__(self, num_classes=200, patch=8, D=192, H=3, depth=12, ratio=4):
        super().__init__()
        n = (64 // patch) ** 2
        self.patch = nn.Conv2d(3, D, patch, patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(D))
        self.pos = nn.Parameter(torch.zeros(n + 1, D))
        self.blocks = nn.ModuleList([ViTBlock(D, H, ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(D, eps=1e-6)
        self.head = nn.Linear(D, num_classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)                         # [B, 64, D]，token 顺序 = 行优先
        x = torch.cat([self.cls.expand(x.shape[0], 1, -1).to(x.dtype), x], 1) + self.pos.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


def build(arch, num_classes=200):
    return {"vgg16": lambda: VGG16(num_classes), "resnet18": lambda: ResNet(BasicBlock, 1, num_classes),
            "resnext26": lambda: ResNet(Bottleneck, 4, num_classes), "vit_tiny": lambda: ViTTiny(num_classes)}[arch]()


def load_tinytensor_init(model, path):
    """npz 中卷积权重是 NHWC 的 [Cout, KH, KW, Cin/g]，PyTorch 需要 [Cout, Cin/g, KH, KW]"""
    data = np.load(path)
    state = model.state_dict()
    for name in data.files:
        w = torch.from_numpy(data[name])
        if w.dim() == 4:
            w = w.permute(0, 3, 1, 2)
        assert state[name].shape == w.shape, (name, state[name].shape, w.shape)
        state[name].copy_(w)
    params = {n for n, _ in model.named_parameters()}
    assert params == set(data.files), params ^ set(data.files)


# ---------------- 数据：与 TinyTensor 的 load_batch kernel 完全相同的增强 ----------------
M32 = 0xFFFFFFFF


def hash32(x):
    """lowbias32，用 int64 运算再截到 32 位"""
    x = x ^ (x >> 16)
    x = (x * 0x7feb352d) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846ca68b) & M32
    return x ^ (x >> 16)


class GpuData:
    def __init__(self, images, labels, batch, dtype, channels_last, augment, seed):
        self.images = images                    # [N, H, W, 3] uint8 (cuda)
        self.labels = labels.long()
        N, H, W, _ = images.shape
        self.H, self.W, self.batch, self.augment, self.seed = H, W, batch, augment, seed
        dev = images.device
        self.arange_b = torch.arange(batch, device=dev)
        self.hh = torch.arange(H, device=dev).view(1, H, 1)
        self.ww = torch.arange(W, device=dev).view(1, 1, W)
        self.mean = torch.tensor(C.MEAN, device=dev).view(1, 1, 1, 3)
        self.inv_std = 1.0 / torch.tensor(C.STD, device=dev).view(1, 1, 1, 3)
        self.dtype, self.channels_last = dtype, channels_last

    def get(self, order, bstep, gstep):
        """bstep：本 epoch 的 batch 序号；gstep：全局步数（决定增强的随机数）。都是 GPU 上的 long 标量"""
        H, W, B, pad = self.H, self.W, self.batch, C.PAD
        idx = order.index_select(0, bstep * B + self.arange_b).long()
        if self.augment:
            b = self.arange_b
            rnd = hash32(self.seed ^ hash32((gstep * 0x9E3779B9 + b) & M32))
            rng = 2 * pad + 1
            dy = (rnd % rng - pad).view(B, 1, 1)
            dx = ((rnd >> 8) % rng - pad).view(B, 1, 1)
            flip = ((rnd >> 16) & 1).view(B, 1, 1).bool()
            sh = self.hh + dy                                      # [B, H, 1]
            sw = torch.where(flip, W - 1 - self.ww, self.ww) + dx  # [B, 1, W]
        else:
            sh, sw = self.hh.expand(B, H, 1), self.ww.expand(B, 1, W)
        valid = ((sh >= 0) & (sh < H)) & ((sw >= 0) & (sw < W))   # [B, H, W]
        flat = (idx.view(B, 1, 1) * H + sh.clamp(0, H - 1)) * W + sw.clamp(0, W - 1)
        x = self.images.view(-1, 3).index_select(0, flat.reshape(-1)).view(B, H, W, 3)
        x = torch.where(valid.unsqueeze(-1), x.float() * (1.0 / 255.0), 0.0)
        x = ((x - self.mean) * self.inv_std).permute(0, 3, 1, 2)     # NHWC 内存 = channels_last
        if not self.channels_last:
            x = x.contiguous()
        return x, self.labels.index_select(0, idx)


class TorchTrainer:
    def __init__(self, args, data):
        self.args, dev = args, "cuda"
        torch.cuda.init()
        torch.empty(1, device=dev)
        free0, total = torch.cuda.mem_get_info()
        self.mem_base = total - free0
        tx, ty, vx, vy = data
        self.n_train, self.n_val = tx.shape[0], vx.shape[0]
        # fp32 默认用 NCHW（cuDNN 的 fp32 卷积在 NCHW 下更快），bf16 用 channels_last（tensor core）
        cl = args.memory_format == "channels_last" or (args.memory_format == "auto" and args.dtype == "bf16")
        self.channels_last = cl
        self.train_data = GpuData(torch.from_numpy(tx).to(dev), torch.from_numpy(ty).to(dev), args.batch,
                                  args.dtype, cl, True, C.aug_seed(args.seed))
        self.val_data = GpuData(torch.from_numpy(vx).to(dev), torch.from_numpy(vy).to(dev), C.EVAL_BATCH,
                                args.dtype, cl, False, 0)
        self.model = build(args.arch, C.NUM_CLASSES).to(dev)
        load_tinytensor_init(self.model, C.init_path(args.arch, args.seed))
        if cl:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.params = list(self.model.parameters())

        s = C.schedule(args, self.n_train // args.batch)
        self.sched = s
        self.lr = torch.tensor(s["base_lr"], device=dev)
        if args.optimizer == "sgd":
            self.opt = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, nesterov=True, weight_decay=args.wd,
                                       fused=True)
        else:
            self.opt = torch.optim.AdamW(self.params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=args.wd, fused=True, capturable=True)
        self.order = torch.zeros(self.n_train, dtype=torch.int32, device=dev)
        self.bstep = torch.zeros((), dtype=torch.long, device=dev)
        self.gstep = torch.zeros((), dtype=torch.long, device=dev)
        self.loss_sum = torch.zeros((), device=dev)
        self.correct = torch.zeros((), dtype=torch.long, device=dev)
        self.graphs, self.pool, self.warm = {}, None, False
        self.step_fn = self._step
        if args.compile:
            self.step_fn = torch.compile(self._step, mode=args.compile, dynamic=False)

    def _lr_at(self, t):
        """与 TinyTensor 优化器 kernel 中的 LrAt 相同（t 为从 0 开始的步数，GPU 上的张量）"""
        s = self.sched
        t = t.float()
        warm = s["base_lr"] * (t + 1) / s["warmup_steps"]
        span = max(1, s["total_steps"] - s["warmup_steps"])
        p = ((t - s["warmup_steps"]) / span).clamp(0, 1)
        cos = s["base_lr"] * (s["final_ratio"] + (1 - s["final_ratio"]) * 0.5 * (1 + torch.cos(math.pi * p)))
        return torch.where(t < s["warmup_steps"], warm, cos)

    def _forward_backward(self, x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.args.dtype == "bf16"):
            logits = self.model(x)
        loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        return loss.detach(), logits.detach()

    def _step(self):
        x, y = self.train_data.get(self.order, self.bstep, self.gstep)
        self.bstep += 1
        self.gstep += 1
        loss, logits = self._forward_backward(x, y)
        self.lr.copy_(self._lr_at(self.gstep - 1))
        self.opt.step()
        self.opt.zero_grad(set_to_none=False)
        self.loss_sum += loss * y.shape[0]
        self.correct += (logits.argmax(1) == y).sum()

    def train_steps(self, steps):
        self.model.train()
        if not self.args.graph:
            for _ in range(steps):
                self.step_fn()
            return
        if not self.warm:
            # 在旁路 stream 上先正常执行几步：cuDNN benchmark 选算法、优化器状态初始化，之后才能录制
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(min(3, steps)):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.warm, steps = True, steps - min(3, steps)
        while steps > 0:
            k = self.args.chunk if steps >= self.args.chunk else steps
            if k not in self.graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    for _ in range(k):
                        self._step()
                self.pool = g.pool()
                self.graphs[k] = g
            self.graphs[k].replay()
            steps -= k

    def begin_epoch(self, order):
        self.order.copy_(torch.from_numpy(order))
        self.bstep.zero_()

    def train_stats(self):
        loss, correct = self.loss_sum.item(), self.correct.item()
        self.loss_sum.zero_()
        self.correct.zero_()
        return loss, correct

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        order = torch.arange(self.n_val, dtype=torch.int32, device="cuda")
        loss, correct = torch.zeros((), device="cuda"), torch.zeros((), dtype=torch.long, device="cuda")
        zero = torch.zeros((), dtype=torch.long, device="cuda")
        for i in range(self.n_val // C.EVAL_BATCH):
            x, y = self.val_data.get(order, zero + i, zero)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.args.dtype == "bf16"):
                logits = self.model(x).float()
            loss += F.cross_entropy(logits, y, reduction="sum")
            correct += (logits.argmax(1) == y).sum()
        return loss.item() / self.n_val, correct.item() / self.n_val

    def sync(self):
        torch.cuda.synchronize()

    def peak_memory_mb(self):
        free, total = torch.cuda.mem_get_info()
        return int((total - free - self.mem_base) / 2**20)


def extra_args(p):
    p.add_argument("--compile", default=None, help="torch.compile 的 mode：default / max-autotune-no-cudagraphs / ...")
    p.add_argument("--memory-format", default="auto", choices=["auto", "channels_last", "contiguous"])


if __name__ == "__main__":
    import sys
    graph_flag = "--graph" in sys.argv
    if graph_flag:
        sys.argv.remove("--graph")
    args = C.parse_args(extra_args)
    args.graph = graph_flag          # PyTorch 默认 eager；--graph 时手动录 CUDA Graph
    assert not (args.graph and args.compile), "--graph 与 --compile 二选一"
    data = C.load_data(args.train_limit)
    trainer = TorchTrainer(args, data)
    del data
    if args.compile:
        key, name = "torch_compile_" + args.compile, "PyTorch {} compile({})".format(torch.__version__, args.compile)
    else:
        mode = "graph" if args.graph else "eager"
        key, name = "torch_" + mode, "PyTorch {} ({})".format(torch.__version__, mode)
    C.run(trainer, args, dict(key=key, name=name))
