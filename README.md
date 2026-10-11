**English** | [简体中文](README.zh-CN.md)

# pku-aiprogramming

Homework for course Programming in Artificial Intelligence, Peking University in 2023, and what grew out of it.

The course work builds a small deep-learning framework, **TinyTensor**, step by step: a CUDA tensor class, layers,
Python bindings, automatic differentiation, optimizers, and finally MNIST training. That is **Part I (basics)**.
**Part II (extensions)** pushes the same framework much further: tensor-core convolutions, BatchNorm, attention,
bf16 mixed precision, CUDA Graphs and hand-fused kernels, enough to train VGG16, ResNet18, ResNeXt26 and ViT-Tiny on
ImageNet-1k at 64x64 on an A100, and to compare speed and accuracy against PyTorch step by step.

## Please do not directly copy my work for your study!

## Layout

```
tinytensor/              the framework: one Python package, organized like PyTorch
  tensor.py              Tensor (a node of the computation graph) and the basic operators
  autograd/              reverse-mode automatic differentiation
  nn/                    Module, Parameter and the layers
    functional.py        stateless operators (F.conv2d, F.linear, F.batch_norm, ...)
    intrinsic.py         fused layers (LinearReLU, ConvReLUMaxPool2d, ConvBNAct2d, ...)
    init.py              parameter initialization
  optim/                 SGD, Adam, AdamW, learning-rate schedules
  utils/data.py          data loaders that keep the dataset on the GPU
  cuda.py                synchronize, memory info, CUDA Graph capture and replay
  models/                MLP / CNN (MNIST), VGG16 / ResNet18 / ResNeXt26 / ViT-Tiny
  csrc/                  C++/CUDA sources: tensor, layers, and csrc/nn (tensor-core conv, BN, LN, attention, bf16)
  _C/                    the three compiled extension modules (written by build_linux.sh)
build_linux.sh           builds tinytensor/csrc into tinytensor/_C with nvcc
tests/                   operator tests, PyTorch comparisons, end-to-end step matching
experiments/
  mnist/                 Task 3: MLP / CNN on MNIST, PyTorch baseline, profiler, bench.sh
  imagenet/              Part II: ImageNet64 / Tiny ImageNet training, PyTorch baseline, profilers, conv micro-benchmark
basics/                  Part I: the course work, HW1-HW5 and final-project Tasks 1-2 (see basics/README.md)
results/                 raw json of every number in this README, plus the plotting scripts
```

## The package

TinyTensor is used the way PyTorch is:

```python
import tinytensor as tt
from tinytensor import nn, optim
import tinytensor.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

x, y = tt.tensor(images), tt.tensor(labels)     # numpy -> GPU
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
print(loss.numpy())                             # the only point where the GPU is synchronized
```

How the pieces divide the work:

- **`nn.functional` (`F`) holds the operators.** Each function is one node of the computation graph, with a
  forward and a backward that call one or two CUDA kernels. It has no state: weights are passed in.
- **`nn` holds the layers.** A layer is a `Module` that owns its `Parameter`s and buffers (for example BatchNorm's
  running statistics) and calls `F` in `forward`. Both are needed for the same reason as in PyTorch: `F.relu` or
  `F.max_pool2d` have nothing to own, while `nn.Conv2d` must own a weight that the optimizer can find.
- **`nn.intrinsic` holds the fused layers**, as `torch.nn.intrinsic` does. `nni.LinearReLU` is `nn.Linear` followed
  by `F.relu`, computed by one kernel; `nni.ConvReLUMaxPool2d` is convolution + ReLU + 2x2 max-pool in one kernel;
  `nni.ConvBNAct2d` is convolution, then BatchNorm + residual add + ReLU in one kernel. Each fused layer has the same
  parameters and the same result as its unfused spelling, so fusion is a choice of layer class, not a second set of
  operators. The models take a `fused` / `fuse` flag that switches between the two spellings.
- **There is one set of operators, with two kernel families behind it, selected by memory format** - the same idea
  as `memory_format` in PyTorch:

  | | `tt.contiguous_format` (default) | `tt.channels_last` |
  |---|---|---|
  | Layout | NCHW | NHWC |
  | Kernels | the course kernels (`myLayer`): direct and im2col convolution, fp32 | `myNN`: tensor-core convolution, grouped convolution, BatchNorm, LayerNorm, attention; fp32 or bf16 |
  | Used by | MNIST MLP / CNN | VGG16, ResNet18, ResNeXt26, ViT-Tiny |

  `model.to(memory_format=tt.channels_last)` selects the family and the physical weight layout;
  `model.to(tt.bfloat16)` turns on mixed precision. `F.conv2d`, `F.linear`, `F.relu`, `F.max_pool2d` and
  `F.cross_entropy` work in both; BatchNorm, LayerNorm, attention, GELU and grouped convolution exist only in
  `channels_last`, and `F.conv_relu_max_pool2d` only in `contiguous_format`. An operator raises
  `NotImplementedError` when its family has no kernel for the request.

Where TinyTensor deliberately differs from PyTorch:

- **Parameters are created lazily.** A layer only declares shapes and initializers; the values are generated on the
  GPU at the first use of the model, so `model.to(memory_format=...)` and `model.manual_seed(seed)` can come first.
- **`model.flatten_parameters()`** puts all parameters into one contiguous buffer. A `fused=True` optimizer then
  updates every parameter with a single kernel (which also computes the learning rate from a step counter on the GPU
  and writes the bf16 copy of the weights), and `optimizer.zero_grad()` is one memset.
- **Data loaders live on the GPU.** `utils.data.TensorLoader` and `ImageLoader` upload the dataset once; `next()` is
  one kernel that gathers the batch (and crops, flips and normalizes images).
- **`tt.cuda.GraphRunner`** captures consecutive training steps into one CUDA Graph and replays it. This works
  because a whole step (batch, forward, backward, update, learning rate) runs without the CPU reading anything back.
- **In `channels_last`, `nn.CrossEntropyLoss` accumulates** the loss sum and the number of correct predictions on the
  GPU; `criterion.statistics()` reads and clears them, typically once per epoch.
- **Not supported:** weight sharing (a parameter may be used by one operator per forward pass), convolution bias,
  and CPU execution of the layer operators.

## Environment and build

Everything local runs in WSL2 on an RTX 4060 Laptop. The A100 runs use Colab.

| Item | Local (WSL2) | A100 (Colab) |
|---|---|---|
| GPU | RTX 4060 Laptop, 8 GB | A100-SXM4-80GB |
| OS | Ubuntu 20.04.6 in WSL2, Windows 11 host | Colab runtime |
| Compiler | gcc 11.5, CUDA toolkit 12.8 | CUDA 13 |
| Python | 3.12, numpy, pybind11, pytest, matplotlib | same |
| PyTorch | 2.14.1+cu126, torchvision 0.29.1 | 2.11.0+cu130 |

Setup from scratch, inside WSL (the NVIDIA driver comes from Windows; install only the `cuda-toolkit-12-8` package):

```bash
python3.12 -m venv ~/tt-venv
~/tt-venv/bin/pip install numpy pybind11 pytest matplotlib torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Build TinyTensor from the repository root. The script needs only nvcc, numpy and the pybind11 headers. It compiles
`tinytensor/csrc` and writes the extension modules `myTensor`, `myLayer` and `myNN` into `tinytensor/_C/`:

```bash
PYTHON=~/tt-venv/bin/python NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_linux.sh
```

After that `import tinytensor` works from the repository root (or with the root on `PYTHONPATH`); the scripts in
`experiments/` and `tests/` add it themselves.

From Windows, call WSL through PowerShell: `wsl -d Ubuntu-20.04 -- bash script.sh`. Git-Bash rewrites `$var` and
`/mnt/...` paths before they reach WSL, so put multi-line commands in a script file.

# Part I: basics

Five homeworks and a three-task final project. [`basics/README.md`](basics/README.md) describes each assignment and how
to run it. Every item runs locally (RTX 4060 Laptop, WSL2, the environment above):

| Item | What it is | Result |
|---|---|---|
| HW1 | LeNet on CIFAR-10 in PyTorch, SGD momentum 0 / 0.7 / 0.8 / 0.9 | 10 epochs, test accuracy 45.99% / 61.66% / 64.57% / 65.06% |
| HW2 | CUDA tensor class and layers (FC, convolution, max-pool, softmax, cross-entropy), forward and backward | 10 C++ self-tests run, 4 / 4 comparisons against PyTorch pass |
| HW3 | the same tensor and layers exposed to Python with pybind11 | 4 / 4 layer tests pass; MNIST loads into GPU tensors |
| HW4 | operators, topological sort and reverse-mode autodiff in Python | 22 / 22 tests pass |
| HW5 | SGD and Adam for a two-layer network on MNIST | 20 epochs, test error 2.77% (SGD) and 2.94% (Adam) |
| Task 1 | CNN on MNIST in PyTorch | 10 epochs, test accuracy 98.61%, 104.9 s |
| Task 2 | the same CNN with `nn.DataParallel` (one GPU, so this measures overhead) | plain 98.64% in 104.7 s; `nn.DataParallel` 98.66% in 105.2 s |
| Task 3 | train on MNIST with the hand-written framework | the next section |

## Task 3: TinyTensor on MNIST

An MLP (784-100-10) and a CNN (two 3x3 conv + ReLU + 2x2 max-pool stages with
16 and 32 channels, then a 128-unit hidden layer) are trained for 10 epochs with SGD, lr 0.1, batch 100, seed 0.
PyTorch trains the same network from the same initial weights with the same data order.

Run inside `experiments/mnist`:

```bash
python ../../tests/test_ops.py          # operator checks, CNN finite differences, fused == unfused, CUDA Graph == eager
python mnist_mlp.py --epochs 10         # TinyTensor MLP (CUDA Graph by default, --no-graph for op-by-op)
python mnist_cnn.py --epochs 10         # TinyTensor CNN
python torch_baseline.py --arch cnn     # same model / init / data order in PyTorch (eager)
python torch_baseline.py --arch cnn --graph                     # PyTorch + manual CUDA Graph
python torch_baseline.py --arch cnn --compile reduce-overhead   # PyTorch + torch.compile (also: default)
python profile_step.py --framework tiny --arch cnn --graph      # per-kernel GPU time + GPU timeline via torch.profiler (CUPTI)
```

Common options: `--optimizer sgd|adam --lr --batch --seed --chunk --data <MNIST raw dir> --save <json>`.
`bench.sh` runs all twelve configurations back to back in one environment, and `results/mnist/plot_results.py`
draws the chart from the saved json files:

```bash
PYTHON=~/tt-venv/bin/python DATA=<MNIST raw dir> EPOCHS=10 SAVE=<repo>/results/mnist/rtx4060 ./bench.sh
python ../../results/mnist/plot_results.py rtx4060
```

The models are `tinytensor.models.MLP` and `tinytensor.models.CNN`; `experiments/mnist/tiny_trainer.py` is the
training loop around them. TinyTensor initializes the weights itself: Kaiming uniform from the CUDA `random` kernel
(`nn.init.kaiming_uniform_`). The TinyTensor scripts export the initial weights to `results/mnist/init/<arch>_seed<seed>.npz`, and
`torch_baseline.py` loads the same file. Run the TinyTensor script first.

### Results

s/epoch is the mean of epochs 2-10, because epoch 1 includes CUDA Graph capture or torch.compile compilation.

**The framework comparison should be read from the A100 table.** The laptop's GPU power state depends on the
Windows power mode, so its timings are only comparable within one back-to-back run. The laptop table was measured
with the power mode set to performance.

**RTX 4060 Laptop, WSL2** (`results/mnist/rtx4060/`):

| Model | Mode | s/epoch | Test acc |
|---|---|---|---|
| MLP | **TinyTensor + CUDA Graph** | **0.032** | 97.59% |
| MLP | TinyTensor op-by-op | 0.074 | 97.59% |
| MLP | PyTorch + CUDA Graph | 0.047 | 97.61% |
| MLP | PyTorch eager | 0.327 | 97.61% |
| MLP | PyTorch torch.compile (default) | 0.447 | 97.61% |
| MLP | PyTorch torch.compile (reduce-overhead) | 0.429 | 97.61% |
| CNN | **TinyTensor + CUDA Graph** | **0.286** | 98.86% |
| CNN | TinyTensor op-by-op | 0.322 | 98.91% |
| CNN | PyTorch + CUDA Graph | 0.353 | 98.86% |
| CNN | PyTorch eager | 0.656 | 98.74% |
| CNN | PyTorch torch.compile (default) | 0.647 | 98.78% |
| CNN | PyTorch torch.compile (reduce-overhead) | 0.459 | 98.83% |

TinyTensor with CUDA Graph is the fastest configuration for both models: 1.5x faster than PyTorch with a manual CUDA
Graph on the MLP and 1.2x faster on the CNN. The ranking is the same as on the A100.

![mnist rtx4060](results/mnist/compare_rtx4060.png)

**A100** (`results/mnist/a100/`):

| Model | Mode | s/epoch | Test acc |
|---|---|---|---|
| MLP | **TinyTensor + CUDA Graph** | **0.040** | 97.59% |
| MLP | TinyTensor op-by-op | 0.109 | 97.59% |
| MLP | PyTorch + CUDA Graph | 0.060 | 97.61% |
| MLP | PyTorch eager | 0.537 | 97.61% |
| MLP | PyTorch torch.compile (default) | 0.795 | 97.60% |
| MLP | PyTorch torch.compile (reduce-overhead) | 0.693 | 97.60% |
| CNN | **TinyTensor + CUDA Graph** | **0.197** | 98.88% |
| CNN | TinyTensor op-by-op | 0.212 | 98.87% |
| CNN | PyTorch + CUDA Graph | 0.219 | 98.74% |
| CNN | PyTorch eager | 0.901 | 98.80% |
| CNN | PyTorch torch.compile (default) | 1.079 | 98.79% |
| CNN | PyTorch torch.compile (reduce-overhead) | 0.745 | 98.77% |

This table was measured with the package as it is in this repository (commit `08e4993`, PyTorch 2.11, CUDA 13), after
`tests/test_ops.py` and `tests/test_nn_ops.py` passed on the same machine. A second pass of the twelve configurations
in the same job reproduced every time to within 5% (TinyTensor 0.041 / 0.107 / 0.197 / 0.212 s/epoch). TinyTensor
with CUDA Graph is the fastest configuration for both models: 1.5x faster than PyTorch with CUDA Graph on the MLP and
1.1x on the CNN.

![mnist a100](results/mnist/compare_a100.png)

Both frameworks start from the same init and the same data order, so their accuracy curves overlap. The small
differences between runs come from the order of float `atomicAdd`s in the weight-gradient reductions (TinyTensor) and
from cuDNN's algorithm choice (PyTorch).

One PyTorch-side limitation: for the MLP, capturing 99 or more training steps into one `torch.cuda.CUDAGraph` fails
at replay with `CUDA error: unspecified launch failure` (PyTorch 2.11 on the A100). 50 steps or fewer work, and the
speed is the same for 1, 10, 20 and 50 steps per graph, so `torch_baseline.py` caps the MLP at 50. The cause was not
found.

### How the training step runs on the GPU

- Tensors stay on the GPU between ops; only the loss is read back, and only when asked for. The training set is
  uploaded once and the shuffled order once per epoch. Each step picks its batch on the GPU from a step counter that
  also lives on the GPU. `gather_xy` is one kernel that gathers x and y and advances the counter.
- Fused kernels:
  - conv + ReLU + 2x2 max-pool. The forward stores a 2-bit argmax. The backward is a gather with no atomics, and it
    also zeroes the weight gradient.
  - FC + bias + ReLU.
  - softmax + cross-entropy in one kernel.
  - one multi-tensor SGD update.
- Small-channel convolutions use direct CUDA kernels. The 3x3 case is specialized at compile time, and each thread
  computes several output channels. The weight gradient accumulates all taps in registers and reduces with warp
  shuffles. Larger convolutions fall back to im2col + strided-batched GEMM.
- Gradients that nothing needs (e.g. w.r.t. the input images) are skipped.
- Consecutive training steps (`--chunk`, default 100) are captured into one CUDA Graph and replayed.

Kernels per training step:

| Model | TinyTensor | PyTorch + CUDA Graph |
|---|---|---|
| CNN | 20 | 45 |
| MLP | 12 | 25 |

### torch.compile on MNIST

On the A100, torch.compile in `default` mode is slower than plain eager PyTorch on both models, and `reduce-overhead`
is slower than eager on the MLP and 1.2x faster than eager on the CNN. Neither comes close to a hand-captured CUDA
Graph. A step here is only 0.1-0.4 ms of GPU work, so the CPU cost of launching it decides the speed.

- `default`: Inductor fuses the elementwise ops, but the compiled forward/backward still runs through the autograd
  engine and many Triton launches.
- `reduce-overhead`: the CUDA-Graph trees add per-call bookkeeping (`cudagraph_mark_step_begin`, input copies). On
  the MLP that costs more than the whole step.
- Compilation itself takes 7-10 s, which is why epoch 1 is much slower.

### GPU utilization on the laptop

These observations come from `profile_step.py` (CUPTI timeline, no extra synchronization inside the step) and from
`nvidia-smi` sampling every 200 ms.

- **Inside a step, the GPU never waits.** With CUDA Graph, in both frameworks, a kernel is running for more than 93%
  of the step's GPU time span. Gaps between kernels are under 2 µs.
- **Between graphs, WSL submission is the limit.** Under WSL2, each graph node costs about 4 µs of CPU time at launch.
  Capturing more steps into one graph therefore saves no CPU time, and every kernel removed from a step saves wall
  time directly. That is why the CNN step went from 50 to 28 to 20 kernels: the fused kernels, one kernel for the
  batch gather and the counter, the weight-gradient memset folded into the pool backward, and a cuBLAS workspace of 0
  (which disables split-K and its extra `splitKreduce` kernels; `TT_CUBLAS_WS=<bytes>` re-enables it).
- **Clocks depend on the power mode on this laptop.** Under the Windows "Balanced" power mode the driver caps the
  card's power and the same code runs slower. Compare frameworks only within one back-to-back run, as `bench.sh`
  does.

# Part II: extensions

The same framework, extended in `tinytensor/csrc/nn` with the `channels_last` kernel family: NHWC convolution on bf16
tensor cores, grouped convolution, BatchNorm, LayerNorm, attention, AdamW, label smoothing, Mixup and CutMix. Four networks are trained on ImageNet-1k
downsampled to 64x64 (1000 classes, 1.28 M training images, 50k validation images) on one A100.

## Setup

- **Networks:** defined in `tinytensor/models/` and mirrored layer by layer in
  `experiments/imagenet/torch_imagenet.py`.

  | Network | Params | Notes |
  |---|---|---|
  | VGG16-BN | 15.2 M | 13 conv + BN + ReLU, 5 max-pools, global average pool, one FC |
  | ResNet18 | 11.7 M | 3x3 stride-1 stem + 2x2 max-pool, 4 stages of basic blocks |
  | ResNeXt26 (32x4d) | 15.4 M | bottleneck blocks [2, 2, 2, 2], 32-group 3x3 convolutions |
  | ViT-Tiny | 5.6 M | 8x8 patches (64 tokens + cls), D=192, 3 heads, 12 pre-norm blocks |

- **Training:**
  - Batch 128, bf16 mixed precision. Random crop (pad 4) and horizontal flip, both done on the GPU.
  - CNNs: SGD with momentum, lr 0.1, weight decay 5e-5, 1 warmup epoch, then cosine decay, 60 epochs.
  - ViT-Tiny: AdamW, lr 1e-3, weight decay 0.05, 5 warmup epochs, then cosine decay, 100 epochs.
  - Regularization, all four networks: label smoothing 0.1, Mixup and CutMix (each step uses one of the two with
    equal probability). Both frameworks implement them identically, from the same per-step random numbers.
- **Same starting point:**
  - TinyTensor initializes the weights and exports them; PyTorch loads that file.
  - Both use the same shuffled order and the same augmentation seeds.
  - `tests/test_imagenet_match.py` checks that the first steps of the two frameworks agree. The test
    fails if the relative loss difference in steps 0-2 reaches 1e-4 in fp32 or 1e-2 in bf16.
- **Precision:**
  - bf16 is mixed precision. PyTorch uses `torch.autocast` with channels_last.
  - TinyTensor keeps fp32 master weights and fp32 BN/LN statistics. Convolutions and matmuls run on bf16 tensor cores
    (`mma.sync`).
- **Timing:** the GPU is synchronized only at the start and end of each timed region. Loss and accuracy are
  accumulated on the GPU and read back once per epoch.
- **What is compared:** TinyTensor (CUDA Graph) runs the whole schedule. PyTorch runs the same schedule but stops
  after epoch 10 (`--stop-epoch 10`), in eager mode and with `torch.compile` (default mode), so the two frameworks
  can be compared epoch by epoch over the first 10 epochs. ViT-Tiny is compared against `torch.compile` only.

## Results on ImageNet64 (A100, bf16, batch 128)

Time per epoch is the median of epochs 2 and later. It covers the training steps of the epoch; the validation pass
is not timed.

| Network | TinyTensor | PyTorch eager | PyTorch torch.compile | TinyTensor vs torch.compile | TinyTensor final top-1 |
|---|---|---|---|---|---|
| ViT-Tiny | **77.4 s** | - | 198.0 s | 2.56x faster | 56.54% (100 epochs) |
| ResNet18 | **74.9 s** | 144.7 s | 104.2 s | 1.39x faster | 56.00% (60 epochs) |
| ResNeXt26 | 149.3 s | 188.9 s | **147.8 s** | 1% slower | 62.55% (60 epochs) |
| VGG16 | 108.9 s | 113.4 s | **83.7 s** | 30% slower | 57.61% (60 epochs) |

torch.compile's first epoch also pays for compilation: 236 s (ViT-Tiny), 136 s (ResNet18), 185 s (ResNeXt26) and
112 s (VGG16).

After 10 epochs of the same schedule, the two frameworks are at the same point:

| Network | Train loss at epoch 10: TinyTensor / eager / compile | Val top-1 at epoch 10: TinyTensor / eager / compile |
|---|---|---|
| ViT-Tiny | 5.1562 / - / 5.1573 | 29.41% / - / 29.52% |
| ResNet18 | 4.8094 / 4.8103 / 4.8113 | 35.37% / 36.16% / 35.31% |
| ResNeXt26 | 4.6147 / 4.6147 / 4.6149 | 40.81% / 40.42% / 40.55% |
| VGG16 | 4.8095 / 4.8099 / 4.8105 | 35.45% / 35.55% / 35.44% |

Validation top-1. Top row: the first 10 epochs, all three configurations on the same schedule. Bottom row: the full
TinyTensor runs.

![imagenet64 training](results/imagenet64_a100_training.png)

Time per epoch:

![imagenet64 time](results/imagenet64_a100_time.png)

What the numbers say:

- **Accuracy is the same in both frameworks.** Training loss at epoch 10 agrees to the fourth significant digit.
  Validation accuracy differs by less than one point, which is the run-to-run noise of bf16 rounding.
- **TinyTensor is clearly faster on ViT-Tiny and ResNet18.**
- **TinyTensor ties torch.compile on ResNeXt26 and loses on VGG16.** VGG16 is a plain stack of wide 3x3
  convolutions, so its time is almost entirely convolution kernels, and cuDNN's are faster than TinyTensor's
  `ConvTc` (see the throughput table below). This is the main open speed item.
- **Batch 128 favours TinyTensor.** At this batch size and image size a PyTorch step is short enough that the CPU
  cost of launching it matters, and TinyTensor replays a CUDA Graph instead. In the batch-256 throughput table
  below, where GPU kernels dominate, torch.compile is ahead on all three CNNs. PyTorch with a manual CUDA Graph or
  `reduce-overhead` was not measured in these training runs.
- **All four accuracy curves are still rising at the last epoch.** The schedules are short for ImageNet.

Wall time of the full TinyTensor runs: ResNet18 1 h 17 min, VGG16 1 h 51 min, ResNeXt26 2 h 32 min, ViT-Tiny
2 h 12 min. Raw results: `results/imagenet64_a100/*.json`; charts: `results/plot_imagenet64.py`.

### Choosing the CNN hyperparameters

ResNet18, TinyTensor, ImageNet64, top-1 on the validation set:

| lr | Weight decay | Regularization | Epochs | Val top-1 |
|---|---|---|---|---|
| 0.1 | 5e-4 | none | 40 | 53.08% |
| 0.025 | 5e-5 | none | 16 | 50.46% |
| 0.05 | 5e-5 | none | 16 | 51.81% |
| 0.1 | 5e-5 | none | 16 | 53.41% |
| 0.1 | 5e-5 | label smoothing + Mixup + CutMix | 60 | 56.00% |

- Weight decay 5e-4, the usual choice for small datasets, is too strong here: 16 epochs at 5e-5 already beat 40
  epochs at 5e-4.
- At 16 epochs without regularization the training accuracy is 68% against 53% on validation, so the longer runs
  use regularization.

### Reproduce

`prepare_imagenet64.py --parquet <downloaded data dir> --out ~/data/imagenet64` converts the 64x64 ImageNet-1k
parquet files to `.npy`. Then, inside `experiments/imagenet`:

```bash
REG="--label-smoothing 0.1 --mixup --cutmix"
python imagenet_train.py --dataset imagenet64 --arch resnet18 --dtype bf16 --epochs 60 --lr 0.1 --wd 5e-5 $REG
python torch_imagenet.py --dataset imagenet64 --arch resnet18 --dtype bf16 --epochs 60 --lr 0.1 --wd 5e-5 $REG --stop-epoch 10
python torch_imagenet.py --dataset imagenet64 --arch resnet18 --dtype bf16 --epochs 60 --lr 0.1 --wd 5e-5 $REG --stop-epoch 10 --compile default
python imagenet_train.py --dataset imagenet64 --arch vit_tiny --dtype bf16 --epochs 100 $REG
python ../../tests/test_nn_ops.py                                   # every channels_last kernel vs PyTorch
python ../../tests/test_imagenet_match.py --arch vit_tiny --dtype bf16   # first steps match PyTorch
python profile_imagenet.py --framework tiny --arch resnext26 --dtype bf16      # per-kernel GPU time
```

Run the TinyTensor script first: it exports the initial weights that the PyTorch script loads. Results are written
to `results/runs/imagenet/`.

## Throughput at batch 256 (A100, ms per step, lower is better)

Measured with `--bench 50 --batch 256 --train-limit 51200` on Tiny ImageNet (same 64x64 input, 200 classes), with no
regularization. All PyTorch columns and the TinyTensor eager column were measured in one session; the TinyTensor +
CUDA Graph column is from a later session after more kernel work (see the log below).

| Network | dtype | TinyTensor + CUDA Graph | TinyTensor eager | PyTorch eager | PyTorch + CUDA Graph | PyTorch torch.compile (best mode) |
|---|---|---|---|---|---|---|
| VGG16 | bf16 | 19.3 | 21.6 | 18.7 | 18.2 | **14.2** |
| ResNet18 | bf16 | 12.6 | 13.6 | 13.4 | 12.6 | **10.0** |
| ResNeXt26 | bf16 | 32.6 | 34.4 | 32.4 | 31.4 | **22.6** |
| ViT-Tiny | bf16 | **14.1** | 14.6 | 27.8 | 20.2 | 19.6 |
| VGG16 | fp32 | 158.1 | 159.4 | 92.5 | 91.6 | **88.9** |
| ResNet18 | fp32 | 84.9 | 85.6 | 60.9 | 60.5 | **58.5** |
| ResNeXt26 | fp32 | **121.7** | 122.8 | 138.9 | 136.5 | 125.7 |
| ViT-Tiny | fp32 | 53.3 | 53.9 | 55.8 | 54.6 | **47.3** |

The three PyTorch modes:

- **eager:** each op launches one cuDNN / cuBLAS / ATen kernel from Python.
- **CUDA Graph:** the same kernels, captured once and replayed. This removes the CPU launch cost.
- **torch.compile:** Dynamo captures the graph and Inductor fuses elementwise ops and reductions into Triton
  kernels. Convolutions and GEMMs still go to cuDNN / cuBLAS. The table shows whichever of `default` and
  `max-autotune-no-cudagraphs` was faster.

TinyTensor has no compiler. Its fusions are written by hand, and it captures its CUDA Graph explicitly. Its fair
counterpart is therefore torch.compile, and the main comparison is against the fastest PyTorch mode.

Where TinyTensor stands at batch 256:

- **ViT-Tiny, bf16: TinyTensor is the fastest, 1.4x faster than torch.compile.**
  - One kernel computes the whole attention for one (batch, head) in shared memory: Q·Kᵀ, softmax, then ·V.
  - LayerNorm takes a single pass.
  - Bias + GELU are fused into the GEMM epilogue.
- **CNNs, bf16: about the same as PyTorch eager and CUDA Graph, but 1.25-1.45x slower than torch.compile.**
  - The remaining gap is the convolutions: cuDNN's implicit-GEMM kernels are still faster than TinyTensor's
    `ConvTc`.
  - Example, a VGG layer with 64 channels: forward 0.22 ms (cuDNN) vs 0.36 ms (TinyTensor); weight gradient 0.36 ms
    vs 0.61 ms.
- **CNNs, fp32: slower, except ResNeXt26.**
  - In fp32, cuDNN switches to Winograd (`scudnn_winograd_128x128`) and FFT (`fft2d_r2c`, `gcgemm`) convolutions.
    These do far fewer FLOPs than an implicit GEMM.
  - TinyTensor's fp32 convolution is a plain implicit GEMM on CUDA cores.
  - ResNeXt26 is dominated by grouped convolutions, where cuDNN has no such shortcut, and TinyTensor's direct
    grouped kernels win.

```bash
python imagenet_train.py --arch resnet18 --dtype bf16 --batch 256 --bench 50 --train-limit 51200          # TinyTensor
python torch_imagenet.py --arch resnet18 --dtype bf16 --batch 256 --bench 50 --train-limit 51200 --graph
python torch_imagenet.py --arch resnet18 --dtype bf16 --batch 256 --bench 50 --train-limit 51200 --compile default
python imagenet_train.py --arch resnet18 --dtype bf16 --batch 256 --bench 50 --train-limit 51200 --no-fuse
```

`prepare_tiny_imagenet.py <tiny-imagenet-200 dir>` writes the `.npy` files these commands read from
`~/data/tiny-imagenet-200`.

### Fusion ablation (`--no-fuse`)

With `--no-fuse` the models are built from the unfused layers (`fuse=False`): BatchNorm only normalizes, and the
residual add, ReLU and GELU each run as a separate kernel. The math is unchanged, and all eight `test_imagenet_match.py` checks pass in this mode too.

Fused vs. unfused, measured back to back in one session (commit 67e2107):

| Network | dtype | fused (ms/step) | unfused (ms/step) | slowdown | peak memory, fused → unfused |
|---|---|---|---|---|---|
| VGG16 | bf16 | 22.97 | 25.01 | +9% | 2702 → 3254 MB |
| ResNet18 | bf16 | 14.31 | 15.84 | +11% | 2394 → 2842 MB |
| ResNeXt26 | bf16 | 35.11 | 41.05 | +17% | 4862 → 6606 MB |
| ViT-Tiny | bf16 | 14.36 | 15.42 | +7% | 2734 → 2746 MB |
| ResNeXt26 | fp32 | 123.05 | 129.82 | +5.5% | 8596 → 12064 MB |

- The other fp32 networks slow down by 1.5-2.5%.
- Fusion matters more in bf16. The convolutions are fast there, so the memory-bound elementwise passes take a larger
  share of the step.
- Fusion also saves memory: the BN output before ReLU and the result of the add are never stored.

### How the bf16 step got faster

Each change was chosen from the per-kernel GPU time (`profile_imagenet.py`, CUPTI) and then measured on the A100.
Times are ms/step at batch 256.

| Change | VGG16 bf16 | ResNet18 bf16 | ResNeXt26 bf16 |
|---|---|---|---|
| Baseline (tensor-core `ConvTc`, scalar BN kernels) | 31.6 | 20.1 | 57.7 |
| BatchNorm: 8 channels per thread, 16-byte loads, no 64-bit modulo per element, larger grids | 22.95 | 14.31 | 35.10 |
| Max-pool fwd/bwd and gradient add: 8 channels per thread, 16-byte loads | 21.53 | 13.49 | 34.10 |
| `ConvTc`: fast division for im2col indices, split-K, 256x128 forward tiles; BN backward recomputes the ReLU mask | 19.27 | 12.55 | 32.60 |

**BatchNorm.** Before this change, BN was the biggest cost in bf16.

- It took about 33 ms of ResNeXt26's 58 ms per step. PyTorch's BN takes about 11 ms.
- The old kernels read 2 bytes per load, and computed `i % C` in 64 bits for every element.
- After the change, BN takes about 11 ms per step in ResNeXt26.

**Convolution (`conv_tc.cu`).**

- bf16 tensor cores via `mma.sync`, multi-stage `cp.async` pipelines, and double-buffered fragments.
- A tile configuration chosen per layer shape (`TT_TC_CFG=small|auto|big`).
- Forward kernels with 256-row tiles stage their output through shared memory, so each warp writes full 16-byte
  vectors. The same staging made dgrad and small layers slower (register spills), so it is enabled only for the
  forward pass.
- Grouped convolutions with 4, 8 or 16 channels per group also run on tensor cores (`conv_grouped.cu`). This was
  added after the last row of the table and has no batch-256 measurement.

**What is left.** The convolution data-gradient and weight-gradient kernels against cuDNN, which is what the VGG16
result above shows, and BatchNorm fusion into the convolution epilogue.

Notes:

- **Stem dgrad.** The 3→64 stem convolution needs no input gradient, so no dgrad kernel runs for it in training. The
  slow stem dgrad that `bench_conv.py` reports never runs in a real step.
- **Graph warm-up in the benchmark.** The timed region is split into CUDA Graphs of `--chunk` steps. Every graph
  length used in the timed region is captured during warm-up.

### Earlier run: Tiny ImageNet, 30 epochs

Before ImageNet64, the four networks were trained on Tiny ImageNet (200 classes, 100k training images) on the same
A100 for 30 epochs without regularization (weight decay 5e-4 for the CNNs). TinyTensor uses CUDA Graph, PyTorch is
eager, top-1 on the validation set:

| Network | dtype | TinyTensor | PyTorch |
|---|---|---|---|
| VGG16 | bf16 | 58.56% | 59.66% |
| ResNet18 | bf16 | 59.74% | 59.96% |
| ResNeXt26 | bf16 | 63.75% | 63.83% |
| ViT-Tiny | bf16 | 41.13% | 41.25% |
| ResNet18 | fp32 | 59.30% | 59.84% |

Raw results and the throughput json files: `results/tiny_imagenet_a100/`; chart: `results/plot_tiny_imagenet.py`.
