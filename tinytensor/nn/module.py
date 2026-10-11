"""
Module：层和模型的基类，用法与 torch.nn.Module 相同（属性赋值即注册、parameters()、train() / eval()、to()）
"""
import numpy as np

from .. import _state
from ..tensor import tf
from .parameter import Parameter, ParameterArena, arenas_of, zero_grad


class Module:
    # 第 i 个参数的随机种子是 seed * seed_stride + i（i 按注册顺序计数）
    seed_stride = 1000

    def __init__(self):
        d = self.__dict__
        d["_children"] = {}     # 名字 -> Parameter / Module / None，按注册顺序
        d["_buffers"] = {}      # 不参与训练的状态（例如 BN 的滑动平均），GPU 张量
        d["training"] = True
        d["memory_format"] = _state.contiguous_format
        d["dtype"] = _state.float32
        d["_seed"] = None
        d["_built"] = False

    def __setattr__(self, name, value):
        children = self.__dict__.get("_children")
        if children is not None and (isinstance(value, (Parameter, Module)) or name in children):
            children[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name, data):
        self._buffers[name] = data
        object.__setattr__(self, name, data)

    # ---------- 遍历 ----------
    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self._children.items():
            if isinstance(child, Module):
                yield from child.named_modules(prefix + ("." if prefix else "") + name)

    def modules(self):
        return [m for _, m in self.named_modules()]

    def named_parameters(self, prefix=""):
        """按注册顺序深度优先遍历（参数与子模块混排，所以名字的顺序就是声明的顺序）"""
        for name, child in self._children.items():
            full = prefix + ("." if prefix else "") + name
            if isinstance(child, Parameter):
                yield full, child
            elif isinstance(child, Module):
                yield from child.named_parameters(full)

    def parameters(self):
        self._build()
        return [p for _, p in self.named_parameters()]

    def named_buffers(self):
        for prefix, m in self.named_modules():
            for name, data in m._buffers.items():
                yield prefix + ("." if prefix else "") + name, data

    # ---------- 分配与初始化 ----------
    def manual_seed(self, seed):
        """指定这个模型的初始化种子（缺省用 tinytensor.manual_seed 设置的全局种子）"""
        self.__dict__["_seed"] = seed
        return self

    def _build(self):
        if self._built:
            return
        seed = self._seed if self._seed is not None else _state.seed
        for i, (_, p) in enumerate(self.named_parameters()):
            p.materialize(seed * self.seed_stride + i)
        for m in self.modules():
            m.__dict__["_built"] = True
        for arena in arenas_of(p for _, p in self.named_parameters()):
            arena.sync_shadow()

    def flatten_parameters(self):
        """把全部参数放进一块连续显存（见 ParameterArena）；融合的优化器（fused=True）需要它"""
        params = [p for _, p in self.named_parameters()]
        if params and params[0].arena is None:
            ParameterArena(params)
            self.__dict__["_built"] = False
        self._build()
        return self

    def to(self, dtype=None, memory_format=None):
        """
        memory_format : contiguous_format（NCHW，课程部分的 fp32 kernel）或 channels_last（NHWC，tensor core kernel）。
                        它同时决定了权重的物理布局，所以只能在参数分配之前设置
        dtype         : float32 或 bfloat16。bfloat16 表示混合精度：fp32 主权重之外再维护一份 bf16 副本
        """
        if dtype in (_state.contiguous_format, _state.channels_last):
            dtype, memory_format = None, dtype
        for m in self.modules():
            if memory_format is not None:
                m.__dict__["memory_format"] = memory_format
            if dtype is not None:
                m.__dict__["dtype"] = dtype
        for _, p in self.named_parameters():
            if memory_format is not None:
                channels_last = memory_format == _state.channels_last
                if p.cached_data is not None and channels_last != p.channels_last and p._shape != p._shape_cl:
                    raise RuntimeError("memory_format must be set before the parameters are allocated")
                p.channels_last = channels_last
            if dtype is not None:
                p.low_precision = dtype == _state.bfloat16
        if dtype is not None:
            for arena in arenas_of(p for _, p in self.named_parameters()):
                arena.sync_shadow()
        return self

    def train(self, mode=True):
        for m in self.modules():
            m.__dict__["training"] = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        zero_grad(self.parameters())

    # ---------- 保存 / 加载 ----------
    def state_dict(self):
        """名字 -> numpy 数组（参数与 buffer），数组是物理布局"""
        self._build()
        state = {name: p.numpy() for name, p in self.named_parameters()}
        state.update((name, data.numpy()) for name, data in self.named_buffers())
        return state

    def load_state_dict(self, state):
        self._build()
        for name, p in self.named_parameters():
            p.raw.copy_from(tf(np.ascontiguousarray(state[name], np.float32), "gpu"), 0)
        for name, data in self.named_buffers():
            if name in state:
                data.copy_from(tf(np.ascontiguousarray(state[name], np.float32), "gpu"), 0)
        for arena in arenas_of(p for _, p in self.named_parameters()):
            arena.sync_shadow()

    def num_parameters(self):
        return sum(int(np.prod(p.shape)) for _, p in self.named_parameters())

    # ---------- 前向 ----------
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        if not self._built:
            self._build()
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._list = []
        for m in modules:
            self.append(m)

    def append(self, module):
        setattr(self, str(len(self._list)), module)
        self._list.append(module)
        return self

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def forward(self, x):
        for m in self._list:
            x = m(x)
        return x


class ModuleList(Sequential):
    """只负责登记子模块，没有 forward"""

    def __init__(self, modules=()):
        super().__init__(*modules)

    def forward(self, *args):
        raise NotImplementedError("ModuleList has no forward(); iterate over it")
