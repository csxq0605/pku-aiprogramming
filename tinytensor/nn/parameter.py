"""
Parameter：模型参数，以及把一个模型的全部参数放进一块连续显存的 ParameterArena

- 参数是计算图之外的叶子：算子把它当作属性持有，反向时把梯度直接写进参数自己的梯度缓冲，
  计算图只连接激活值。因此一个参数在一次前向里只能被一个算子使用（不支持权重共享）
- 声明与分配分开：构造层时只记录形状和初始化方式，第一次使用模型（或调用 model.parameters() /
  model.flatten_parameters()）时才在 GPU 上生成数值。这样 model.to(memory_format=...) 可以在分配前决定物理布局
"""
import numpy as np

from .._C import myNN
from ..tensor import Tensor, tf

tb = myNN.Tensor_bf16
ALIGN = 64  # arena 里每个参数的起始位置按 64 个元素（256 字节）对齐，方便向量化 / cp.async 访存


class Parameter(Tensor):
    """
    shape      : contiguous_format（NCHW）下的物理形状
    shape_cl   : channels_last（NHWC）下的物理形状，缺省与 shape 相同
    init       : "zeros" / "ones" / ("uniform", bound)，也可以之后用 nn.init 里的函数修改
    data       : 直接给定数值（numpy 数组或 GPU 张量），此时不再初始化
    """

    def __init__(self, shape=None, *, shape_cl=None, init="zeros", fan_in=None, fan_out=None, data=None,
                 channels_last=False):
        if data is not None and isinstance(data, np.ndarray):
            data = tf(np.ascontiguousarray(data, dtype=np.float32), "gpu")
        self._init(None, [], cached_data=data, requires_grad=True)
        if data is not None:
            shape = tuple(data.get_shape())
        self._shape = tuple(shape)
        self._shape_cl = tuple(shape_cl) if shape_cl is not None else self._shape
        self.init, self.fan_in, self.fan_out = init, fan_in, fan_out
        self.channels_last = channels_last
        self.low_precision = False      # True：算子遇到 bf16 输入时使用 bf16 副本（由 arena 和优化器维护）
        self.shadow = None              # bf16 副本
        self.arena = None
        self.seed = None
        self._slot = None               # arena 预留的 (权重视图, 梯度视图)
        self._g = None                  # 梯度（GPU 张量）
        self._accumulate = False        # True：梯度是累加型缓冲，需要 zero_grad() 清零

    # ---------- 形状与数值 ----------
    @property
    def shape(self):
        return self._shape_cl if self.channels_last else self._shape

    @property
    def raw(self):
        """fp32 主权重（GPU 张量）"""
        if self.cached_data is None:
            raise RuntimeError("parameter is not initialized yet: call model.parameters() or run the model once")
        return self.cached_data

    def storage(self, dtype):
        """算子实际读取的权重：bf16 输入用 bf16 副本，否则用 fp32 主权重"""
        if dtype != "bf16":
            return self.raw
        if self.shadow is None:
            raise RuntimeError("bf16 input needs a bf16 copy of the weights: call model.to(tinytensor.bfloat16) "
                               "and model.flatten_parameters()")
        return self.shadow

    def materialize(self, seed):
        """在 GPU 上生成初始值；seed 决定均匀分布的随机数，同一 seed 可复现"""
        if self.cached_data is not None:
            return
        self.seed = seed
        shape = list(self.shape)
        data = self._slot[0] if self._slot is not None else tf(shape, "gpu").zeros()
        if self.init == "ones":
            data.copy_from(tf(np.ones(shape, np.float32), "gpu"), 0)
        elif self.init != "zeros":
            kind, bound = self.init
            assert kind == "uniform", self.init
            data.random(-float(bound), float(bound), seed)
        self.cached_data = data
        if self._slot is not None:
            self._g, self._accumulate = self._slot[1], True

    # ---------- 梯度 ----------
    def grad_buffer(self):
        """累加型算子（NHWC 的 kernel 把梯度加进缓冲）使用；优化器的 zero_grad() 负责清零"""
        if self._g is None or not self._accumulate:
            self._g = tf(list(self.shape), "gpu").zeros()
            self._accumulate = True
        return self._g

    def set_grad(self, data):
        """覆盖型算子（NCHW 的 kernel 每次写出新的梯度张量）使用"""
        self._g, self._accumulate = data, False

    @property
    def grad(self):
        return None if self._g is None else Tensor.make_const(self._g)

    @grad.setter
    def grad(self, value):
        self._g = None if value is None else value.realize_cached_data()


class ParameterArena:
    """
    把一组参数放进一块扁平显存（权重 / 梯度 / bf16 副本各一块），每个参数是其中的一个视图：
    优化器用一个 kernel 更新全部参数，梯度清零只是一次 memset，bf16 副本由优化器 kernel 在更新时顺便写出
    """

    def __init__(self, params):
        self.params = list(params)
        offset = 0
        for p in self.params:
            assert p.arena is None, "parameter already belongs to an arena"
            p.offset = offset
            offset += (int(np.prod(p.shape)) + ALIGN - 1) // ALIGN * ALIGN
        self.numel = offset
        self.flat_w = tf([offset], "gpu").zeros()
        self.flat_g = tf([offset], "gpu").zeros()
        self.flat_shadow = None
        for p in self.params:
            shape = list(p.shape)
            w, g = self.flat_w.view(p.offset, shape), self.flat_g.view(p.offset, shape)
            p.arena = self
            if p.cached_data is None:
                p._slot = (w, g)        # 还没初始化：之后直接在视图上生成
            else:
                w.copy_from(p.cached_data, 0)
                p.cached_data, p._g, p._accumulate = w, g, True

    def sync_shadow(self):
        """把 fp32 主权重整体转成 bf16 副本（之后由优化器 kernel 维护）"""
        if not any(p.low_precision for p in self.params):
            return
        if self.flat_shadow is None:
            self.flat_shadow = tb([self.numel], "gpu").zeros()
            for p in self.params:
                p.shadow = self.flat_shadow.view(p.offset, list(p.shape))
        myNN.cast(self.flat_w, self.flat_shadow)

    def zero_grad(self):
        self.flat_g.zeros()


def arenas_of(params):
    """这组参数用到的 arena（去重，保持顺序）"""
    return list({id(p.arena): p.arena for p in params if p.arena is not None}.values())


def zero_grad(params):
    """
    清零梯度：arena 里的参数整块 memset 一次；单独存放的累加型梯度逐个清零；
    覆盖型梯度（NCHW 的 kernel 每次写出新的张量）不需要清零
    """
    for arena in arenas_of(params):
        arena.zero_grad()
    for p in params:
        if p.arena is None and p._accumulate:
            p._g.zeros()
