"""GPU 相关的工具：同步、显存信息、把连续的训练 step 录成 CUDA Graph 重放"""
from ._C import myNN, myTensor


def synchronize():
    """等待 GPU 上已提交的计算全部完成（计时用）"""
    myTensor.synchronize()


def mem_info():
    """(空闲显存, 总显存)，字节"""
    return myNN.mem_info()


def release_pool():
    """把显存池里缓存的空闲块还给驱动（CUDA Graph 的私有池不受影响）"""
    myNN.release_pool()


class GraphRunner:
    """
    重复执行 step_fn（一个完整的训练 step：取 batch、前向、反向、参数更新，全部在 GPU 上、不读回任何数值）。
    enabled=True 时把连续 chunk 个 step 录进同一张 CUDA Graph，之后每 chunk 个 step CPU 只发射一次。

    - 第一个 step 正常执行：cuBLAS 句柄、kernel 属性、优化器状态等惰性初始化不能发生在录制期间
    - 录制期间分配的显存来自私有池，不会被图外的代码复用。录制结束时仍然存活的张量必须一直保持存活
      （否则它的显存会回到普通池、被别的张量占用，而重放时图还会往里写），所以每张图都保存 step_fn 的返回值
      和 keep() 的返回值
    """

    def __init__(self, step_fn, chunk=100, enabled=True, keep=None):
        self.step_fn, self.chunk, self.enabled, self.keep = step_fn, max(1, chunk), enabled, keep
        self.graphs = {}
        self.warm = False

    def _graph(self, k):
        if k not in self.graphs:
            # 录制时 GPU 不执行，录完后由 run() 发射
            myTensor.graph_capture_begin()
            for _ in range(k):
                result = self.step_fn()
            graph = myTensor.graph_capture_end()
            self.graphs[k] = (graph, result, self.keep() if self.keep else None)
        return self.graphs[k]

    def run(self, steps):
        """连续执行 steps 个 step（不同步），返回最后一个 step 的 step_fn 返回值"""
        result = None
        if not self.enabled:
            for _ in range(steps):
                result = self.step_fn()
            return result
        if not self.warm and steps > 0:
            self.warm = True
            result = self.step_fn()
            release_pool()   # 这一步的中间结果已经释放进普通池，录制用的是私有池，先把它们还掉
            steps -= 1
        while steps > 0:
            k = min(self.chunk, steps)
            graph, result, _ = self._graph(k)
            graph.launch()
            steps -= k
        return result
