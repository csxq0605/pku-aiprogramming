[English](README.md) | **简体中文**

# 第一部分：课程作业

五次作业和一个包含三个任务的大作业。它们合起来自底向上搭出一个小型深度学习框架：CUDA 张量、层、Python 绑定、
自动微分、优化器，以及 MNIST 训练。最终得到的框架就是仓库根目录下 [`tinytensor/`](../tinytensor) 的起点。

下面的每个结果都在 RTX 4060 Laptop 的 WSL2 里测得（环境见[主 README](../README.zh-CN.md)）。`<MNIST root>` 是包含
`MNIST/raw/` 的目录；`<CIFAR root>` 是包含 `cifar-10-batches-py/` 的目录。两者缺失时都由 torchvision 下载。

## HW1：用 PyTorch 训练 CIFAR-10（`hw1_cifar10/`）

**任务。** 用 `torch.nn` 和 SGD 在 CIFAR-10 上训练 LeNet，画出 loss 和准确率，并比较不同的动量。作业的后半部分是
第一个带 ReLU 和 sigmoid 的 CUDA 张量类，它是 HW2 代码的前半部分。

```bash
python cifar.py --momentum 0.9 --data <CIFAR root>     # 也可以是 0、0.7、0.8
```

**结果。** 10 个 epoch，学习率 0.006，测试准确率：

| 动量 | 0 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| 测试准确率 | 45.99% | 61.66% | 64.57% | 65.06% |

在这个学习率下，动量越大收敛越快。曲线和 json：`hw1_cifar10/results/`。

## HW2：用 CUDA 写张量和层（`hw2_layers/`）

**任务。** 用 C++/CUDA 写一个 GPU 张量类，以及全连接层、卷积（im2col + GEMM）、最大池化、softmax 和交叉熵的前向
与反向。

```bash
./build_linux.sh          # 构建，运行 C++ 自测，然后与 PyTorch 对照（test_layers.py）
```

**结果。** 10 个 C++ 自测运行；与 PyTorch 的 4 / 4 项 pytest 对照通过。

## HW3：Python 绑定（`hw3_pybind/`）

**任务。** 用 pybind11 把张量类和各层暴露给 Python，实现与 numpy 的互相转换，并在 Python 里把每个层与 PyTorch
对照测试。把 MNIST 读入 GPU 张量。

```bash
./build_linux.sh && cd test && pytest test_layer.py && python test_Tensor.py <MNIST root>
```

**结果。** 4 / 4 项层测试通过；MNIST 能读入 GPU 张量。

## HW4：自动微分（`hw4_autodiff/`）

**任务。** 实现每个算子（加、乘、幂、除、转置、reshape、广播、求和、矩阵乘、取负、log、exp、ReLU）的前向计算和
梯度，计算图的拓扑排序，以及在其上的反向模式微分。设计参照 CMU《深度学习系统》课程的 Needle 框架。

```bash
pytest
```

**结果。** 22 / 22 通过。

## HW5：优化器（`hw5_optimizer/`）

**任务。** 在 HW4 的基础上，用手写的 SGD 和 Adam 在 MNIST 上训练一个两层网络，并比较两者。

```bash
python task1_optimizer.py <MNIST root>
```

**结果。** 20 个 epoch，测试错误率 SGD（学习率 0.2）为 2.77%，Adam（学习率 0.001）为 2.94%。两者在大约第 5 个
epoch 之后都开始过拟合（测试 loss 不再下降），Adam 在各 epoch 之间的波动更大。

## 大作业

### Task 1：用 PyTorch 在 MNIST 上训练 CNN（`task1_pytorch_mnist/`）

**任务。** 用 PyTorch 在 MNIST 上训练一个 CNN，给出 loss 和准确率曲线。

```bash
python mnist.py --data <MNIST root>
```

**结果。** 10 个 epoch，测试准确率 98.61%，104.9 s。

### Task 2：数据并行训练（`task2_parallel/`）

**任务。** 用 `nn.DataParallel` 训练同一个 CNN，与单设备的训练比较时间和准确率。

```bash
python mnist.py --data <MNIST root>
```

**结果。** 普通训练 98.64%，104.7 s；`nn.DataParallel` 98.66%，105.2 s。这台机器只有一张 GPU，所以
`nn.DataParallel` 在这里不可能加速：这个对比测到的是它的开销，而这个开销在运行间的噪声范围之内。

### Task 3：用自己的框架在 MNIST 上训练

**任务。** 把 HW2-HW5 合在一起：用手写的张量、层、自动微分和优化器在 MNIST 上训练一个 MLP 和一个 CNN，并与
PyTorch 比较。

代码就是框架本身 [`tinytensor/`](../tinytensor)，由 [`experiments/mnist/`](../experiments/mnist) 驱动。命令和结果见
主 README 的 "Task 3" 一节。
