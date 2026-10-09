# pku-aiprogramming
Homework for course Programming in Artificial Intelligence, Peking University in 2023.



## Please do not directly copy my work for your study!



## Build

Please download pybind11 to the  folder tinytensor and run setup.py to build. Referring to the PDF task3 for more information.

Task3 is built and benchmarked on Linux / WSL. Run the build inside `Task3/Tinytensor`. The script only needs nvcc,
numpy and pybind11 headers, and it writes the modules to `optimizer/`. `PYTHON` selects the interpreter the modules are
built for:

```bash
PYTHON=~/tt-venv/bin/python NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_linux.sh
```

`setup.py` (PyTorch `CUDAExtension`) is kept as an alternative build.

### WSL environment used for all numbers below

| Item | Version / setting |
|---|---|
| Host | Windows 11, RTX 4060 Laptop (8 GB), NVIDIA driver 616.92 installed on Windows; WSL reaches it through `/usr/lib/wsl/lib` |
| WSL | `Ubuntu-20.04` (20.04.6), WSL2 kernel 5.15.167.4-microsoft, 32 CPU threads, ~7 GB RAM given to WSL |
| Compiler | gcc 11.5, CUDA toolkit 12.8 (`/usr/local/cuda-12.8/bin/nvcc`); no driver is installed inside WSL |
| Python | venv `~/tt-venv`: Python 3.12.7, numpy, pybind11, matplotlib, torch 2.14.1+cu126, triton 3.8.0 |
| Data | MNIST raw files (`train-images-idx3-ubyte` ...), passed with `--data <dir>` |

To set it up from scratch, run these inside WSL:

```bash
python3.12 -m venv ~/tt-venv
~/tt-venv/bin/pip install numpy pybind11 matplotlib torch --index-url https://download.pytorch.org/whl/cu126
# CUDA toolkit only (the driver comes from Windows): install NVIDIA's "WSL-Ubuntu" cuda-toolkit-12-8 package
cd Task3/Tinytensor && PYTHON=~/tt-venv/bin/python NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_linux.sh
cd optimizer && ~/tt-venv/bin/python test_ops.py
```

From Windows, call WSL through PowerShell: `wsl -d Ubuntu-20.04 -- bash script.sh`. Git-Bash rewrites `$var` and
`/mnt/...` paths before they reach WSL, so put multi-line commands in a script file.
TinyTensor and PyTorch are always compared in the same venv and the same WSL session.

## Run Task3

Run inside `Task3/Tinytensor/optimizer`:

```bash
python test_ops.py                      # operator checks, CNN finite differences, fused == unfused, CUDA Graph == eager
python mnist_mlp.py --epochs 10         # TinyTensor MLP (CUDA Graph by default, --no-graph for op-by-op)
python mnist_cnn.py --epochs 10         # TinyTensor CNN
python torch_baseline.py --arch cnn     # same model / init / data order in PyTorch (eager)
python torch_baseline.py --arch cnn --graph                     # PyTorch + manual CUDA Graph
python torch_baseline.py --arch cnn --compile reduce-overhead   # PyTorch + torch.compile (also: default)
python profile_step.py --framework tiny --arch cnn --graph      # per-kernel GPU time + GPU timeline via torch.profiler (CUPTI)
```

Common options: `--optimizer sgd|adam --lr --batch --seed --chunk --data <MNIST raw dir> --save <json>`.
`../bench_wsl.sh` runs all twelve configurations in one environment. `Task3/results/plot_results.py` then draws
`compare.png` from the saved json files:

```bash
PYTHON=~/tt-venv/bin/python DATA=<MNIST raw dir> EPOCHS=10 SAVE=<repo>/Task3/results/json ./bench_wsl.sh
```

TinyTensor initializes the weights itself: Kaiming uniform from the CUDA `random` kernel (`tiny_nn.kaiming_uniform`).
The TinyTensor scripts export the initial weights to `Task3/results/init/<arch>_seed<seed>.npz`, and
`torch_baseline.py` loads the same file. Run the TinyTensor script first.

How the training step runs on the GPU:

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

## Results

Setup: RTX 4060 Laptop, WSL Ubuntu 20.04, Python 3.12, PyTorch 2.14.1+cu126 in the same venv. 10 epochs, SGD lr=0.1,
batch 100, seed 0. s/epoch is the mean of epochs 2-10, because epoch 1 includes CUDA Graph capture or torch.compile
compilation. `bench_wsl.sh` ran all twelve configurations back to back:

| Model | Mode | s/epoch | Test acc |
|---|---|---|---|
| MLP | **TinyTensor + CUDA Graph** | **0.043** | 97.59% |
| MLP | TinyTensor op-by-op | 0.092 | 97.59% |
| MLP | PyTorch + CUDA Graph | 0.066 | 97.61% |
| MLP | PyTorch eager | 0.337 | 97.61% |
| MLP | PyTorch torch.compile (default) | 0.510 | 97.61% |
| MLP | PyTorch torch.compile (reduce-overhead) | 0.489 | 97.61% |
| CNN | **TinyTensor + CUDA Graph** | **0.100** | 98.09% |
| CNN | TinyTensor op-by-op | 0.166 | 98.03% |
| CNN | PyTorch + CUDA Graph | 0.249 | 98.05% |
| CNN | PyTorch eager | 0.711 | 98.00% |
| CNN | PyTorch torch.compile (default) | 0.812 | 97.94% |
| CNN | PyTorch torch.compile (reduce-overhead) | 0.574 | 98.08% |

![compare](Task3/results/compare.png)

Both frameworks start from the same init and the same data order, so their accuracy curves overlap. The small
differences between runs come from the order of float `atomicAdd`s in the weight-gradient reductions (TinyTensor) and
from cuDNN's algorithm choice (PyTorch).

### torch.compile

On this workload torch.compile is slower than plain eager PyTorch. A step here is only 0.1-0.4 ms of GPU work, so
the CPU cost of launching it decides the speed.

- `default`: Inductor fuses the elementwise ops, but the compiled forward/backward still runs through the autograd
  engine and many Triton launches.
- `reduce-overhead`: the CUDA-Graph trees add per-call bookkeeping (`cudagraph_mark_step_begin`, input copies), and that
  costs more than the whole step.
- Compilation itself takes 7-10 s, which is why epoch 1 is much slower.
- Compiling the full step with compiled autograd was slower still (about 0.9 s/epoch for the MLP).

The fastest PyTorch variant measured was not a standard torch.compile mode. It took Inductor-compiled kernels
(`max-autotune-no-cudagraphs`) and captured them, together with the optimizer, into a manual CUDA Graph: MLP 0.046
s/epoch, CNN 0.20 s/epoch. That is still slower than TinyTensor + CUDA Graph, and because it is not part of the
standard API it is left out of the table.

### GPU utilization: stable and efficient?

These numbers come from `profile_step.py` (CUPTI timeline, no extra synchronization inside the step) and from
`nvidia-smi` sampling every 200 ms.

- **Inside a step, the GPU never waits.** With CUDA Graph, in both frameworks, a kernel is running for more than 93%
  of the step's GPU time span. Gaps between kernels are under 2 µs, and there are no 10-50 µs bubbles.
- **Between graphs, WSL submission is the limit.** Under WSL2, each graph node costs about 4 µs of CPU time at launch.
  Capturing more steps into one graph therefore saves no CPU time, and every kernel removed from a step saves wall
  time directly. That is why the CNN step went from 50 to 28 to 20 kernels. The changes were:
  - the fused kernels;
  - merging the batch gather and the counter into one kernel;
  - folding the weight-gradient memset into the pool backward;
  - setting the cuBLAS workspace to 0, which disables split-K and its extra `splitKreduce` kernels. This alone took
    the CNN from 0.219 to 0.179 ms/step, even though the GEMMs themselves got slightly slower. `TT_CUBLAS_WS=<bytes>`
    re-enables split-K.
- **Per-step time, CNN:**

  | Framework | Wall time | GPU busy |
  |---|---|---|
  | TinyTensor + CUDA Graph | 0.20 ms/step | 0.16 ms/step |
  | PyTorch + CUDA Graph | 0.42 ms/step | almost all of it, mostly cuDNN `wgrad`/`dgrad` |

- **Clocks are not stable on this laptop.**
  - During the benchmark, the SM clock moved between 1.8 and 2.5 GHz (the maximum is 3.1 GHz).
  - The throttle reason was mostly `SW Power Cap` (0x4). The `SW Thermal Slowdown` counter has also accumulated a lot
    of time.
  - Under the Windows "Balanced" power plan, the driver holds the card at about 40 W. The same code measured up to
    1.7x slower during a power-capped period.
  - So compare frameworks only within one back-to-back run, as `bench_wsl.sh` does.
  - For lower and steadier absolute times: switch Windows or the laptop vendor's tool to performance mode, stay on AC
    power, and keep the laptop cool. The code does not change any system settings.
