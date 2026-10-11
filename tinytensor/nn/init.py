"""
参数初始化。数值由 TinyTensor 自己的 CUDA random kernel 生成（均匀分布），可导出给 PyTorch 对照实现加载。
这些函数可以在参数分配之前调用（只记录初始化方式），也可以在之后调用（立即重新生成）
"""
import numpy as np


def _set(param, init):
    param.init = init
    if param.cached_data is not None:
        data, seed = param.cached_data, param.seed
        param.cached_data = None
        param._slot = (data.zeros(), param._g)
        param.materialize(seed if seed is not None else 0)
    return param


def uniform_(param, bound):
    """U(-bound, bound)"""
    return _set(param, ("uniform", bound))


def kaiming_uniform_(param):
    """ReLU 增益的 Kaiming 均匀分布：U(-b, b)，b = sqrt(6 / fan_in)，方差 2 / fan_in"""
    return uniform_(param, np.sqrt(6.0 / param.fan_in))


def xavier_uniform_(param):
    """U(-b, b)，b = sqrt(6 / (fan_in + fan_out))"""
    return uniform_(param, np.sqrt(6.0 / (param.fan_in + param.fan_out)))


def fan_in_uniform_(param):
    """PyTorch nn.Linear 权重的默认初始化：U(-1 / sqrt(fan_in), 1 / sqrt(fan_in))"""
    return uniform_(param, 1.0 / np.sqrt(param.fan_in))


def std_uniform_(param, std=0.02):
    """标准差为 std 的均匀分布（ViT 的 cls token 与位置编码）"""
    return uniform_(param, std * np.sqrt(3.0))


def zeros_(param):
    return _set(param, "zeros")


def ones_(param):
    return _set(param, "ones")
