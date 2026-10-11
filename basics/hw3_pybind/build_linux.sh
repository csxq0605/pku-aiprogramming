#!/usr/bin/env bash
# 直接用 nvcc 编译 myTensor / myLayer 两个扩展（Linux / WSL），生成到 test/ 下
# 用法: PYTHON=python3 NVCC=nvcc ./build_linux.sh
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
NVCC=${NVCC:-nvcc}
PY_INCLUDE=$($PYTHON -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND11_INCLUDE=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS="-O2 -std=c++17 -Xcompiler -fPIC -I$PY_INCLUDE -I$PYBIND11_INCLUDE -Isrc"
mkdir -p build
for f in Tinytensor.cu TinyTensor_kernels.cu Layers.cu Layers_kernels.cu bind_Tensor.cpp bind_Layer.cpp; do
    $NVCC $FLAGS -x cu -c "src/$f" -o "build/${f%.*}.o"
done
$NVCC -shared build/bind_Tensor.o build/Tinytensor.o build/TinyTensor_kernels.o -o "test/myTensor$EXT_SUFFIX"
$NVCC -shared build/bind_Layer.o build/Layers.o build/Layers_kernels.o build/Tinytensor.o build/TinyTensor_kernels.o     -lcublas -o "test/myLayer$EXT_SUFFIX"
echo "built test/myTensor$EXT_SUFFIX test/myLayer$EXT_SUFFIX"
