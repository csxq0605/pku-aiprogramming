#!/usr/bin/env bash
# 不依赖 PyTorch，直接用 nvcc 把 tinytensor/csrc 编译成 tinytensor/_C 下的三个扩展 myTensor / myLayer / myNN（Linux / WSL）
# 用法: PYTHON=python3 PYBIND11_INCLUDE=/path/to/pybind11/include ./build_linux.sh
set -e
cd "$(dirname "$0")"

SRC=tinytensor/csrc
OUT=tinytensor/_C
PYTHON=${PYTHON:-python3}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-native}
PY_INCLUDE=$($PYTHON -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
if [ -z "$PYBIND11_INCLUDE" ]; then
    PYBIND11_INCLUDE=$($PYTHON -c "import pybind11; print(pybind11.get_include())" 2>/dev/null || echo "pybind11/include")
fi

# CUDA 13 起模板 kernel 的主机端入口默认只在本文件可见，而 Tinytensor.cu 会调用 TinyTensor_kernels.cu 里的模板 kernel
EXTRA=""
NVCC_MAJOR=$($NVCC --version | grep -o "release [0-9]*" | grep -o "[0-9]*$")
if [ "${NVCC_MAJOR:-0}" -ge 13 ]; then EXTRA="-static-global-template-stub=false"; fi
FLAGS="$EXTRA -O3 -std=c++17 --default-stream per-thread -diag-suppress 20281 -arch=$ARCH -Xcompiler -fPIC -I$PY_INCLUDE -I$PYBIND11_INCLUDE -I$SRC"
mkdir -p build/obj/nn

compile() {
    $NVCC $FLAGS -x cu -c "$SRC/$1" -o "build/obj/${1%.*}.o"
}

# 各文件互不依赖，最多 6 个并行编译（WSL 内存有限）；ONLY="xxx.cu ..." 时只重编这些文件
FILES="Tinytensor.cu TinyTensor_kernels.cu Layers.cu Layers_kernels.cu bind_Tensor.cpp bind_Layer.cpp
       nn/conv.cu nn/conv_tc.cu nn/conv_grouped.cu nn/conv_f32.cu nn/conv_stem.cu nn/attention_tc.cu nn/norm.cu nn/misc.cu nn/linear.cu nn/attention.cu nn/bind_nn.cpp"
export NVCC FLAGS SRC
export -f compile
printf "%s\n" ${ONLY:-$FILES} | xargs -P 6 -I{} bash -c 'echo "compiling {}" && compile {}'

$NVCC -shared -arch=$ARCH build/obj/bind_Tensor.o build/obj/Tinytensor.o build/obj/TinyTensor_kernels.o \
    -o "$OUT/myTensor$EXT_SUFFIX"
$NVCC -shared -arch=$ARCH build/obj/bind_Layer.o build/obj/Layers.o build/obj/Layers_kernels.o \
    build/obj/Tinytensor.o build/obj/TinyTensor_kernels.o -lcublas \
    -o "$OUT/myLayer$EXT_SUFFIX"
NN_OBJS=$(ls build/obj/nn/*.o)
$NVCC -shared -arch=$ARCH $NN_OBJS build/obj/Tinytensor.o build/obj/TinyTensor_kernels.o -lcublas -o "$OUT/myNN$EXT_SUFFIX"
echo "built $OUT/myTensor$EXT_SUFFIX $OUT/myLayer$EXT_SUFFIX $OUT/myNN$EXT_SUFFIX"
