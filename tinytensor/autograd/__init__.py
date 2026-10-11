"""反向模式自动微分：计算图节点（function.py）与逆拓扑序的梯度传播（engine.py）"""
from .function import Op, Value  # noqa: F401
from .engine import compute_gradient_of_variables, find_topo_sort  # noqa: F401
