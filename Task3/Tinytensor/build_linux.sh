#!/usr/bin/env bash
# 不依赖 PyTorch，直接用 nvcc 编译 myTensor / myLayer 两个扩展（Linux / WSL）
# 用法: PYTHON=python3 PYBIND11_INCLUDE=/path/to/pybind11/include ./build_linux.sh
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-native}
PY_INCLUDE=$($PYTHON -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
if [ -z "$PYBIND11_INCLUDE" ]; then
    PYBIND11_INCLUDE=$($PYTHON -c "import pybind11; print(pybind11.get_include())" 2>/dev/null || echo "pybind11/include")
fi

FLAGS="-O3 -std=c++17 --default-stream per-thread -diag-suppress 20281 -arch=$ARCH -Xcompiler -fPIC -I$PY_INCLUDE -I$PYBIND11_INCLUDE -Isrc"
mkdir -p build/obj

compile() {
    $NVCC $FLAGS -x cu -c "src/$1" -o "build/obj/${1%.*}.o"
}

for f in Tinytensor.cu TinyTensor_kernels.cu Layers.cu Layers_kernels.cu bind_Tensor.cpp bind_Layer.cpp; do
    echo "compiling $f"
    compile "$f"
done

$NVCC -shared -arch=$ARCH build/obj/bind_Tensor.o build/obj/Tinytensor.o build/obj/TinyTensor_kernels.o \
    -o "optimizer/myTensor$EXT_SUFFIX"
$NVCC -shared -arch=$ARCH build/obj/bind_Layer.o build/obj/Layers.o build/obj/Layers_kernels.o \
    build/obj/Tinytensor.o build/obj/TinyTensor_kernels.o -lcublas \
    -o "optimizer/myLayer$EXT_SUFFIX"
echo "built optimizer/myTensor$EXT_SUFFIX optimizer/myLayer$EXT_SUFFIX"
