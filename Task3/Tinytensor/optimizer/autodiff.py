"""
本文件我们给出进行自动微分的步骤
"""
from typing import List, Dict, Tuple
from basic_operator import Op, Value
import sys
sys.setrecursionlimit(10**8)

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    给定一个节点列表，返回以这些节点结束的拓扑排序列表。
    一种简单的算法是对给定的节点进行后序深度优先搜索(DFS)遍历,
    根据输入边向后遍历。由于一个节点是在其所有前驱节点遍历后才被添加到排序中的，
    因此我们得到了一个拓扑排序。
    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
    

def compute_gradient_of_variables(output_tensor, out_grad, free_graph=False):
    """
    对输出节点相对于 node_list 中的每个节点求梯度。
    将计算结果存储在每个 Variable 的 grad 字段中。
    free_graph=True 时（图像模型用）：一个中间节点的梯度一旦传给它的输入就释放，
    它自己的前向结果也随之释放（逆拓扑序下，用到它的节点都已经处理完），
    只给叶子节点保留 grad，显存峰值与 PyTorch 释放计算图的方式一致。
    """
    # map for 从节点到每个输出节点的梯度贡献列表
    node_to_output_grads_list = {}
    # 我们实际上是在对标量 reduce_sum(output_node) 
    # 而非向量 output_node 取导数。
    # 但这是损失函数的常见情况。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 根据我们要对其求梯度的 output_node，以逆拓扑排序遍历图。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        if node not in node_to_output_grads_list:
            # 没有任何梯度流到该节点（它的所有使用者都不需要它的梯度）
            continue
        autodiff_joints = node_to_output_grads_list[node]
        v_i = autodiff_joints[0]
        for i in range(len(autodiff_joints)):
            if i == 0:
                continue
            v_i = v_i + autodiff_joints[i]
        if node.op is None or not free_graph:
            node.grad = v_i
        if free_graph:
            del node_to_output_grads_list[node]

        if node.op is None:
            continue
        node_grads = node.op.gradient_as_tuple(v_i, node)
        for node_input, node_grad in zip(node.inputs, node_grads):
            if node_grad is None:
                # 算子判定该输入不需要梯度
                continue
            node_to_output_grads_list.setdefault(node_input, [])
            node_to_output_grads_list[node_input].append(node_grad)
        if free_graph:
            del v_i, node_grads
            if hasattr(node.op, "release"):
                node.op.release()
            if node is not output_tensor:
                node.cached_data = None