"""
分层计时（不插入同步）：用 torch.profiler (CUPTI) 采集同一进程内所有 CUDA kernel 的实际执行时间，
TinyTensor 自己的 kernel 与 cuBLAS 也能采到。走的是真实训练路径（begin_epoch + train_batch）。

- 各算子（仅 TinyTensor 逐算子模式）：CPU 时间 = Python 封装、自动微分、发射 kernel；GPU 时间 = 该算子 kernel 在 GPU 时间线上的跨度
- GPU kernel 明细：每个 kernel 每 step 的执行时间
- GPU 占用率 = kernel 总时间 / step 时间；远低于 100% 说明 GPU 在等 CPU

用法: python profile_step.py --framework tiny|torch --arch mlp|cnn [--graph] [--steps 200 --batch 100]
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mnist_common import DEFAULT_DATA_DIR, SGD, parse_mnist


def _annotate(fn, label):
    def wrapper(*args, **kwargs):
        with record_function(label):
            return fn(*args, **kwargs)
    return wrapper


def tiny_setup(arch, use_graph, chunk):
    import myTensor
    import operators
    from tiny_nn import CNN, MLP

    if not use_graph:
        # 给每个算子的前向 / 反向加上 record_function 标记（只是 CPU 侧的区间标记，不会同步 GPU）
        for name in dir(operators):
            cls = getattr(operators, name)
            if isinstance(cls, type) and issubclass(cls, operators.TensorOp) and cls is not operators.TensorOp:
                for method, tag in (("compute", "fwd"), ("gradient", "bwd")):
                    if method in cls.__dict__:
                        setattr(cls, method, _annotate(cls.__dict__[method], "op:{}.{}".format(name, tag)))
    model = (CNN if arch == "cnn" else MLP)(0, use_graph=use_graph, chunk=chunk)
    return model, SGD(0.1), myTensor.synchronize


def torch_setup(arch, use_graph, chunk):
    from torch_baseline import TorchModel
    model = TorchModel(arch, 0, "cuda", use_graph=use_graph, chunk=chunk)
    return model, SGD(0.1), torch.cuda.synchronize


def device_time(evt):
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def self_device_time(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["tiny", "torch"], default="tiny")
    parser.add_argument("--arch", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--graph", action="store_true", help="使用 CUDA Graph")
    parser.add_argument("--chunk", type=int, default=100, help="CUDA Graph 每张图包含的 step 数")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    parser.add_argument("--kernels", type=int, default=15, help="显示最耗时的前 N 个 kernel")
    args = parser.parse_args()

    (X, y), _ = parse_mnist(args.data)
    n, b = args.steps, args.batch
    model, optimizer, sync = (tiny_setup if args.framework == "tiny" else torch_setup)(args.arch, args.graph, args.chunk)
    model.begin_epoch(X, y, np.arange(X.shape[0]))

    def run(first, count):
        model.train_steps(first, count, b, optimizer)

    # 预热：第一个 step 正常执行，随后录好 chunk 个 step 的图，计时区间内不再录制
    warm = 1 + args.chunk if args.graph else 20
    run(0, warm)
    sync()

    # 1) 不开 profiler 的真实 step 时间（只在首尾同步）
    start = time.perf_counter()
    run(warm, n)
    sync()
    wall_ms = (time.perf_counter() - start) / n * 1e3

    # 2) profiler 采集
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run(warm + n, n)
        sync()
    events = prof.key_averages()

    marks = defaultdict(lambda: {"count": 0, "cpu": 0.0, "gpu": 0.0})
    for e in events:
        if e.key.startswith("op:"):
            m = marks[e.key]
            if e.device_type.name == "CPU":
                m["count"], m["cpu"] = e.count, e.cpu_time_total
            else:
                m["gpu"] = device_time(e) or self_device_time(e)

    def is_kernel(e):
        return e.device_type.name == "CUDA" and not e.key.startswith("op:")

    kernels = sorted([e for e in events if is_kernel(e)], key=lambda e: -self_device_time(e))
    kernel_total = sum(self_device_time(e) for e in kernels) / n / 1e3

    name = "TinyTensor" if args.framework == "tiny" else "PyTorch " + torch.__version__
    print("\n######## {} {}  {}  batch={}  steps={} ########".format(
        name, args.arch.upper(), "CUDA Graph x{}".format(args.chunk) if args.graph else "逐算子执行", b, n))
    if marks:
        print("\n== 各算子（每 step 平均，CPU 时间含 Python 封装）==")
        print("  {:<30} {:>7} {:>10} {:>10}".format("", "次/step", "CPU ms", "GPU ms"))
        for k, v in sorted(marks.items(), key=lambda kv: -max(kv[1]["cpu"], kv[1]["gpu"])):
            print("  {:<30} {:>7} {:>10.3f} {:>10.3f}".format(k[3:], v["count"] // n, v["cpu"] / n / 1e3, v["gpu"] / n / 1e3))
    print("\n== GPU kernel（前 {} 个，每 step 平均）==".format(args.kernels))
    for e in kernels[:args.kernels]:
        print("  {:>8.4f} ms  x{:<3d} {}".format(self_device_time(e) / n / 1e3, e.count // n, e.key[:90]))
    print("  共 {} 种 kernel，每 step 约 {} 次 kernel/拷贝".format(len(kernels), sum(e.count for e in kernels) // n))
    print("\n真实 step 时间（无 profiler）: {:.3f} ms  ->  600 step/epoch 约 {:.3f} s".format(wall_ms, wall_ms * 0.6))
    print("GPU kernel 总时间: {:.3f} ms/step  ->  GPU 占用率约 {:.0f}%".format(kernel_total, kernel_total / wall_ms * 100))
    timeline(prof, n)


def timeline(prof, steps):
    """GPU 时间线上相邻 kernel 之间的空隙：短空隙是 kernel 之间的切换开销，长空隙说明 GPU 在等 CPU"""
    spans = sorted((e.time_range.start, e.time_range.end) for e in prof.events()
                   if e.device_type.name == "CUDA" and not e.name.startswith("op:"))
    if not spans:
        return
    gaps, end = [], spans[0][1]
    busy = spans[0][1] - spans[0][0]
    for s, e in spans[1:]:
        if s > end:
            gaps.append(s - end)
            busy += e - s
        elif e > end:
            busy += e - end
        end = max(end, e)
    total = end - spans[0][0]
    print("\n== GPU 时间线（每 step 平均）==")
    print("  跨度 {:.3f} ms：有 kernel 在跑 {:.3f} ms（{:.0f}%），空闲 {:.3f} ms".format(
        total / steps / 1e3, busy / steps / 1e3, busy / total * 100, (total - busy) / steps / 1e3))
    for lo, hi, label in ((0, 2, "< 2 us（kernel 切换）"), (2, 10, "2-10 us"), (10, 50, "10-50 us"),
                          (50, float("inf"), "> 50 us（等 CPU）")):
        g = [x for x in gaps if lo <= x < hi]
        print("  空隙 {:<20} {:>6.1f} 次  共 {:.3f} ms".format(label, len(g) / steps, sum(g) / steps / 1e3))


if __name__ == "__main__":
    main()
