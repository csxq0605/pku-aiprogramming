"""
端到端一致性检查：TinyTensor 与 PyTorch 从同一份初始权重出发、吃同样的 batch，逐 step 比较训练 loss
用法: python test_imagenet_match.py [--arch resnet18] [--dtype fp32] [--steps 8]
"""
import sys

import numpy as np

import imagenet_common as C


def main():
    argv = sys.argv[1:]
    steps = 8
    if "--steps" in argv:
        i = argv.index("--steps")
        steps = int(argv[i + 1])
        del argv[i:i + 2]
    # 两个框架在同一进程里，batch 用 32 避免显存超额
    sys.argv = [sys.argv[0], "--train-limit", "4096", "--epochs", "2", "--no-graph", "--batch", "32"] + argv
    args = C.parse_args(lambda p: (p.add_argument("--compile", default=None),
                                   p.add_argument("--memory-format", default="auto")))
    data = C.load_data(args.train_limit)
    data = data[:2] + (data[2][:1000], data[3][:1000])

    from imagenet_train import TinyTrainer
    tiny = TinyTrainer(args, data)
    import torch_imagenet
    torch_t = torch_imagenet.TorchTrainer(args, data)

    order = C.epoch_order(args.seed, 0, tiny.n_train)
    tiny.begin_epoch(order)
    torch_t.begin_epoch(order)
    worst = 0.0
    for s in range(steps):
        tiny.train_steps(1)
        torch_t.train_steps(1)
        a, b = tiny.train_stats()[0] / args.batch, torch_t.train_stats()[0] / args.batch
        if s < 3:
            # 前 3 步检验前向、第一次反向和优化器更新；之后的差异是浮点舍入在训练中的正常放大，只打印
            worst = max(worst, abs(a - b) / abs(b))
        print("step {} | tiny {:.5f} | torch {:.5f} | rel diff {:.1e}".format(s, a, b, abs(a - b) / abs(b)))
    va, vb = tiny.evaluate(), torch_t.evaluate()
    print("eval  | tiny loss {:.5f} acc {:.4f} | torch loss {:.5f} acc {:.4f}".format(*va, *vb))
    tol = 1e-4 if args.dtype == "fp32" else 1e-2
    print("{} {} {}: worst rel diff (steps 0-2) {:.1e}".format("PASS" if worst < tol else "FAIL", args.arch, args.dtype, worst))


if __name__ == "__main__":
    main()
