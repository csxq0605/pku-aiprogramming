"""
学习率计划。学习率不在 Python 里计算：优化器 kernel 读 GPU 上的全局步数，按计划算出当前学习率，
所以不需要每个 step 调用 scheduler.step()，连续的 step 也可以录进同一张 CUDA Graph
"""
from .._C import myNN


class WarmupCosine:
    """
    线性 warmup（第 t 步为 lr * (t + 1) / warmup_steps），之后余弦衰减到 lr * final_ratio
    step : GPU 上的步数计数器，通常是数据加载器的 loader.step（取 batch 的 kernel 顺便把全局步数加 1）
    """

    def __init__(self, optimizer, warmup_steps, total_steps, final_ratio=0.0, step=None):
        if not optimizer.fused:
            raise NotImplementedError("lr schedules are computed inside the fused optimizer kernel (fused=True)")
        self.optimizer = optimizer
        optimizer.schedule = myNN.LrSchedule(base_lr=optimizer.lr, warmup_steps=warmup_steps, total_steps=total_steps,
                                             final_ratio=final_ratio)
        if step is not None:
            optimizer.step_counter = step
