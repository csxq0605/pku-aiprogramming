#!/usr/bin/env bash
# 在同一个 Python 环境里依次跑 TinyTensor / PyTorch 的各种配置，输出每种配置的稳定 epoch 时间与准确率
# 用法: PYTHON=~/tt-venv/bin/python DATA=/path/to/MNIST/raw EPOCHS=3 ./bench_wsl.sh
set -e
cd "$(dirname "$0")/optimizer"
PYTHON=${PYTHON:-python3}
EPOCHS=${EPOCHS:-3}
DATA=${DATA:-data/MNIST/raw}
SAVE=${SAVE:-}

run() {
    local name=$1; shift
    local save_args=()
    if [ -n "$SAVE" ]; then save_args=(--save "$SAVE/$name.json"); fi
    printf "%-26s " "$name"
    "$PYTHON" "$@" --epochs "$EPOCHS" --data "$DATA" "${save_args[@]}" 2>/dev/null | grep "Test accuracy"
}

for arch in mlp cnn; do
    run "tinytensor_${arch}_graph" "mnist_${arch}.py"
    run "tinytensor_${arch}_eager" "mnist_${arch}.py" --no-graph
    run "pytorch_${arch}_eager"    torch_baseline.py --arch "$arch"
    run "pytorch_${arch}_graph"    torch_baseline.py --arch "$arch" --graph
    # torch.compile：第 1 个 epoch 含编译时间，稳定 epoch 时间取第 2 个 epoch 起
    run "pytorch_${arch}_compile"  torch_baseline.py --arch "$arch" --compile default
    run "pytorch_${arch}_compile_ro" torch_baseline.py --arch "$arch" --compile reduce-overhead
done
