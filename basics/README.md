# Part I: the course work

Five homeworks and a three-task final project. Together they build a small deep-learning framework from the bottom
up: a CUDA tensor, layers, Python bindings, automatic differentiation, optimizers, and MNIST training. The framework
that results is the starting point of [`tinytensor/`](../tinytensor) in the repository root.

Every result below was measured on an RTX 4060 Laptop in WSL2 (environment in the [main README](../README.md)).
`<MNIST root>` is a directory that contains `MNIST/raw/`; `<CIFAR root>` contains `cifar-10-batches-py/`. Both are
downloaded by torchvision if missing.

## HW1: CIFAR-10 in PyTorch (`hw1_cifar10/`)

**Task.** Train LeNet on CIFAR-10 with `torch.nn` and SGD, plot loss and accuracy, and compare momentum values.
The second half of the homework, a first CUDA tensor class with ReLU and sigmoid, is the first half of the HW2 code.

```bash
python cifar.py --momentum 0.9 --data <CIFAR root>     # also 0, 0.7, 0.8
```

**Result.** 10 epochs, lr 0.006, test accuracy:

| Momentum | 0 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| Test accuracy | 45.99% | 61.66% | 64.57% | 65.06% |

At this learning rate more momentum converges faster. Curves and json: `hw1_cifar10/results/`.

## HW2: tensor and layers in CUDA (`hw2_layers/`)

**Task.** Write a GPU tensor class in C++/CUDA and the forward and backward passes of a fully connected layer,
convolution (im2col + GEMM), max pooling, softmax and cross-entropy.

```bash
./build_linux.sh          # builds, runs the C++ self-tests, then compares with PyTorch (test_layers.py)
```

**Result.** 10 C++ self-tests run; 4 / 4 pytest comparisons against PyTorch pass.

## HW3: Python bindings (`hw3_pybind/`)

**Task.** Expose the tensor class and the layers to Python with pybind11, convert to and from numpy, and test each
layer against PyTorch from Python. Load MNIST into GPU tensors.

```bash
./build_linux.sh && cd test && pytest test_layer.py && python test_Tensor.py <MNIST root>
```

**Result.** 4 / 4 layer tests pass; MNIST loads into GPU tensors.

## HW4: automatic differentiation (`hw4_autodiff/`)

**Task.** Implement the forward computation and the gradient of each operator (add, multiply, power, divide,
transpose, reshape, broadcast, sum, matmul, negate, log, exp, ReLU), a topological sort of the computation graph, and
reverse-mode differentiation over it. The design follows the Needle framework of CMU's Deep Learning Systems course.

```bash
pytest
```

**Result.** 22 / 22 pass.

## HW5: optimizers (`hw5_optimizer/`)

**Task.** On top of HW4, train a two-layer network on MNIST with hand-written SGD and Adam, and compare them.

```bash
python task1_optimizer.py <MNIST root>
```

**Result.** 20 epochs, test error 2.77% with SGD (lr 0.2) and 2.94% with Adam (lr 0.001). Both overfit after about
epoch 5 (the test loss stops falling), and Adam is noisier from epoch to epoch.

## Final project

### Task 1: CNN on MNIST in PyTorch (`task1_pytorch_mnist/`)

**Task.** Train a CNN on MNIST with PyTorch and report loss and accuracy curves.

```bash
python mnist.py --data <MNIST root>
```

**Result.** 10 epochs, test accuracy 98.61%, 104.9 s.

### Task 2: data-parallel training (`task2_parallel/`)

**Task.** Train the same CNN with `nn.DataParallel` and compare time and accuracy with the single-device run.

```bash
python mnist.py --data <MNIST root>
```

**Result.** Plain 98.64% in 104.7 s; `nn.DataParallel` 98.66% in 105.2 s. This machine has one GPU, so
`nn.DataParallel` cannot speed anything up here: the comparison measures its overhead, which is within run-to-run
noise.

### Task 3: train on MNIST with your own framework

**Task.** Put HW2-HW5 together: train an MLP and a CNN on MNIST with the hand-written tensor, layers, autodiff and
optimizers, and compare with PyTorch.

The code is the framework itself, [`tinytensor/`](../tinytensor), driven by
[`experiments/mnist/`](../experiments/mnist). Commands and results are in the main README under "Task 3".
