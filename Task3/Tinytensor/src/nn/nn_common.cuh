// 图像模型（VGG / ResNet / ResNeXt / ViT）共用的定义
// 布局约定：激活值 NHWC；卷积核 [Cout, KH, KW, Cin/groups]；全连接权重 [out, in]（与 PyTorch 相同）
// 元素类型 T 为 float（FP32 训练）或 bf16（混合精度：激活 bf16，权重主副本 / 梯度 / 归一化统计量 fp32）
#ifndef NN_COMMON_CUH
#define NN_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "base.h"
#include "TinyTensor.h"

using bf16 = __nv_bfloat16;

template <typename T> __device__ __forceinline__ float ToF(T x);
template <> __device__ __forceinline__ float ToF<float>(float x){ return x; }
template <> __device__ __forceinline__ float ToF<bf16>(bf16 x){ return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T FromF(float x);
template <> __device__ __forceinline__ float FromF<float>(float x){ return x; }
template <> __device__ __forceinline__ bf16 FromF<bf16>(float x){ return __float2bfloat16(x); }

inline void NNCheck(bool ok, const std::string& what){
    if (!ok){
        throw std::invalid_argument(what);
    }
}

template <typename T>
inline void NNCheckGpu(const TinyTensor<T>& t, const std::string& what){
    NNCheck(t.device == "gpu" && t.p_data != nullptr, what + ": tensor must be on gpu");
}

// 网格大小：元素数较多时用固定数量的 block 做 grid-stride 循环
// 当前设备的 SM 数（split-K、常驻 block 数等启发式用）
inline int DeviceSmCount(){
    static int n = [](){ int dev = 0, v = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); return v; }();
    return n;
}

inline int GridFor(size_t n, int block = BLOCK_SIZE){
    size_t blocks = (n + block - 1) / block;
    return (int)std::min<size_t>(std::max<size_t>(blocks, 1), 65535);
}

// 卷积的形状（NHWC）
struct ConvGeom{
    int n, h, w, c;          // 输入
    int k, r, s;             // 输出通道、卷积核高宽
    int p, q;                // 输出高宽
    int pad_h, pad_w, stride_h, stride_w;
    int groups;
};

inline ConvGeom MakeConvGeom(const std::vector<int>& x_shape, const std::vector<int>& w_shape,
                             int pad_h, int pad_w, int stride_h, int stride_w, int groups){
    NNCheck(x_shape.size() == 4 && w_shape.size() == 4, "conv: expect NHWC input and [K,R,S,C/g] weight");
    ConvGeom g;
    g.n = x_shape[0]; g.h = x_shape[1]; g.w = x_shape[2]; g.c = x_shape[3];
    g.k = w_shape[0]; g.r = w_shape[1]; g.s = w_shape[2];
    g.groups = groups;
    NNCheck(groups >= 1 && g.c % groups == 0 && g.k % groups == 0 && w_shape[3] == g.c / groups,
            "conv: channel / group mismatch");
    g.pad_h = pad_h; g.pad_w = pad_w; g.stride_h = stride_h; g.stride_w = stride_w;
    g.p = (g.h + 2 * pad_h - g.r) / stride_h + 1;
    g.q = (g.w + 2 * pad_w - g.s) / stride_w + 1;
    return g;
}

#endif
