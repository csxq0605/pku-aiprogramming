"""
按 kernel 统计一个训练 step 的 GPU 时间（torch.profiler / CUPTI 在进程内采集，TinyTensor 的 kernel 同样能看到）
WSL 下自带的 nsys 2024.6 采不到新驱动的 kernel，所以用这个脚本代替
用法: python profile_imagenet.py --framework tiny --arch resnet18 --dtype bf16 [--steps 10] [--top 30]
"""
import re
import sys
from collections import defaultdict

import torch
from torch.profiler import profile, ProfilerActivity

import imagenet_common as C


def extra(p):
    p.add_argument("--framework", default="tiny", choices=["tiny", "torch"])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--memory-format", default="auto", choices=["auto", "channels_last", "contiguous"])
    p.add_argument("--compile", default=None)


def short(name):
    name = re.sub(r"\(anonymous namespace\)::", "", name)
    name = re.sub(r"void |\(.*$", "", name)
    return name[:100]


def main():
    args = C.parse_args(extra)
    args.graph = False   # 逐 kernel 计时用 eager；CUDA Graph 只省 launch，不改变 kernel 时间
    data = C.load_data(args.train_limit or 20000)
    if args.framework == "tiny":
        from imagenet_train import TinyTrainer as T
    else:
        from torch_imagenet import TorchTrainer as T
    trainer = T(args, data)
    del data
    trainer.begin_epoch(C.epoch_order(args.seed, 0, trainer.n_train))
    trainer.train_steps(10)
    trainer.sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        trainer.train_steps(args.steps)
        trainer.sync()
    agg = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(("Memcpy", "Memset")):
            a = agg[short(e.name)]
            a[0] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            a[1] += 1
    total = sum(v[0] for v in agg.values())
    print("{} {} {} | GPU kernel time {:.2f} ms/step".format(args.framework, args.arch, args.dtype,
                                                              total / 1e3 / args.steps))
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print("{:5.1f}% {:7.2f} ms/step {:5d}/step  {}".format(100 * t / total, t / 1e3 / args.steps,
                                                             n // args.steps, name))


if __name__ == "__main__":
    main()
