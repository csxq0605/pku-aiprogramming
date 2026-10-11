#!/usr/bin/env bash
# 不用 CMake，直接用 nvcc 编译并运行（Linux / WSL）：生成 testdata.txt / cppout.txt 后由 test_layers.py 与 PyTorch 对比
# 用法: NVCC=nvcc PYTHON=python3 ./build_linux.sh
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
PYTHON=${PYTHON:-python3}
mkdir -p build
$NVCC -O2 -std=c++17 -Isrc src/main.cu -o build/TinyTensorProject -lcublas
cd build
echo q | ./TinyTensorProject > run.log
grep -c "passed" run.log
$PYTHON -m pytest -q -p no:cacheprovider ../test_layers.py
