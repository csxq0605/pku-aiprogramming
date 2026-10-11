// NHWC 卷积：隐式 GEMM（不显式展开 im2col，加载 tile 时按下标直接从输入里取）
//   前向   FWD  : M = N*P*Q（输出像素），N = K/g（输出通道），Kdim = R*S*C/g
//   输入梯度 DGRAD: M = N*H*W（输入像素），N = C/g，          Kdim = R*S*K/g（步长 > 1 时不整除的抽头取 0）
//   权重梯度 WGRAD: M = K/g，              N = R*S*C/g，      Kdim = N*P*Q（split-K，atomicAdd 到 fp32 梯度）
// 分组卷积（ResNeXt）的每一组是一个独立的 GEMM，放在 grid.z 上
#include <cstdlib>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

// bf16 且通道数对齐时走 tensor core 版本（conv_tc.cu）；环境变量 TT_CONV_SIMT=1 强制用 SIMT 版本（测试对照用）
template <typename T>
bool UseTensorCore(const ConvGeom& g, int mode){
    static const bool force_simt = std::getenv("TT_CONV_SIMT") != nullptr;
    return std::is_same<T, bf16>::value && !force_simt && ConvTensorCoreSupported(g, mode);
}

// fp32 且通道数对齐时走 conv_f32.cu 的大 tile 版本（同样受 TT_CONV_SIMT 控制）
template <typename T>
bool UseF32(const ConvGeom& g, int mode){
    static const bool force_simt = std::getenv("TT_CONV_SIMT") != nullptr;
    return std::is_same<T, float>::value && !force_simt && ConvF32Supported(g, mode);
}

enum ConvMode { FWD = 0, DGRAD = 1, WGRAD = 2 };

struct ConvArgs{
    ConvGeom g;
    int cg, kg;          // 每组的输入 / 输出通道数
    int gm, gn, gk;      // GEMM 维度
    int k_per_split;     // split-K 时每段的 Kdim 长度
    int splits;
};

// GEMM 的 A(m, kk) 元素
template <typename T, int MODE>
__device__ __forceinline__ float LoadA(const T* __restrict__ a, const ConvArgs& p, int group, int m, int kk){
    const ConvGeom& g = p.g;
    if (MODE == FWD){
        // m -> (n, oh, ow)，kk -> (r, s, c)
        int ow = m % g.q; int t = m / g.q; int oh = t % g.p; int n = t / g.p;
        int c = kk % p.cg; t = kk / p.cg; int s = t % g.s; int r = t / g.s;
        int ih = oh * g.stride_h - g.pad_h + r;
        int iw = ow * g.stride_w - g.pad_w + s;
        if (ih < 0 || ih >= g.h || iw < 0 || iw >= g.w) return 0.f;
        return ToF(a[(((size_t)n * g.h + ih) * g.w + iw) * g.c + group * p.cg + c]);
    }
    else if (MODE == DGRAD){
        // m -> (n, ih, iw)，kk -> (r, s, k)；输出位置 oh = (ih + pad - r) / stride，必须整除
        int iw = m % g.w; int t = m / g.w; int ih = t % g.h; int n = t / g.h;
        int k = kk % p.kg; t = kk / p.kg; int s = t % g.s; int r = t / g.s;
        int th = ih + g.pad_h - r, tw = iw + g.pad_w - s;
        if (th < 0 || tw < 0 || th % g.stride_h || tw % g.stride_w) return 0.f;
        int oh = th / g.stride_h, ow = tw / g.stride_w;
        if (oh >= g.p || ow >= g.q) return 0.f;
        return ToF(a[(((size_t)n * g.p + oh) * g.q + ow) * g.k + group * p.kg + k]);
    }
    else{
        // A(m = 输出通道 j, kk = 输出像素)
        return ToF(a[(size_t)kk * g.k + group * p.kg + m]);
    }
}

// GEMM 的 B(kk, n) 元素
template <typename T, int MODE>
__device__ __forceinline__ float LoadB(const T* __restrict__ b, const ConvArgs& p, int group, int kk, int n){
    const ConvGeom& g = p.g;
    if (MODE == FWD){
        return ToF(b[(size_t)(group * p.kg + n) * p.gk + kk]);
    }
    else if (MODE == DGRAD){
        int k = kk % p.kg; int t = kk / p.kg; int s = t % g.s; int r = t / g.s;
        return ToF(b[(((size_t)(group * p.kg + k) * g.r + r) * g.s + s) * p.cg + n]);
    }
    else{
        // B(kk = 输出像素, n = (r, s, c))：输入 x 按 im2col 取
        int ow = kk % g.q; int t = kk / g.q; int oh = t % g.p; int nn = t / g.p;
        int c = n % p.cg; t = n / p.cg; int s = t % g.s; int r = t / g.s;
        int ih = oh * g.stride_h - g.pad_h + r;
        int iw = ow * g.stride_w - g.pad_w + s;
        if (ih < 0 || ih >= g.h || iw < 0 || iw >= g.w) return 0.f;
        return ToF(b[(((size_t)nn * g.h + ih) * g.w + iw) * g.c + group * p.cg + c]);
    }
}

constexpr int SBM = 64, SBN = 64, SBK = 16;

// 256 线程，每个线程算 4x4 个输出（行 ty + 16i，列 tx + 16j：写回时相邻线程写相邻列，合并访存）
// A/B 在全局内存里沿 Kdim 连续时（A_K_INNER / B_K_INNER），加载时让相邻线程读相邻的 kk
template <typename T, typename OutT, int MODE, bool A_K_INNER, bool B_K_INNER>
__global__ void __launch_bounds__(256) ConvGemmSimt(const T* __restrict__ a, const T* __restrict__ b,
                                                     OutT* __restrict__ out, const ConvArgs p){
    __shared__ float As[2][SBK][SBM + 4];
    __shared__ float Bs[2][SBK][SBN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int m0 = blockIdx.x * SBM, n0 = blockIdx.y * SBN;
    const int group = blockIdx.z / p.splits, split = blockIdx.z % p.splits;
    const int k_begin = split * p.k_per_split;
    const int k_end = min(p.gk, k_begin + p.k_per_split);

    float ra[4], rb[4];
    auto load = [&](int k0){
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            int mm = A_K_INNER ? (tid / SBK + 16 * i) : (tid % SBM);
            int kk = A_K_INNER ? (tid % SBK) : (tid / SBM + 4 * i);
            int m = m0 + mm, k = k0 + kk;
            ra[i] = (m < p.gm && k < k_end) ? LoadA<T, MODE>(a, p, group, m, k) : 0.f;
            int nn = B_K_INNER ? (tid / SBK + 16 * i) : (tid % SBN);
            int kb = B_K_INNER ? (tid % SBK) : (tid / SBN + 4 * i);
            int n = n0 + nn; k = k0 + kb;
            rb[i] = (n < p.gn && k < k_end) ? LoadB<T, MODE>(b, p, group, k, n) : 0.f;
        }
    };
    auto store = [&](int buf){
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            int mm = A_K_INNER ? (tid / SBK + 16 * i) : (tid % SBM);
            int kk = A_K_INNER ? (tid % SBK) : (tid / SBM + 4 * i);
            As[buf][kk][mm] = ra[i];
            int nn = B_K_INNER ? (tid / SBK + 16 * i) : (tid % SBN);
            int kb = B_K_INNER ? (tid % SBK) : (tid / SBN + 4 * i);
            Bs[buf][kb][nn] = rb[i];
        }
    };

    float acc[4][4] = {};
    load(k_begin);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = k_begin; k0 < k_end; k0 += SBK){
        const bool more = k0 + SBK < k_end;
        if (more) load(k0 + SBK);
        #pragma unroll
        for (int kk = 0; kk < SBK; ++kk){
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i){ av[i] = As[buf][kk][ty + 16 * i]; bv[i] = Bs[buf][kk][tx + 16 * i]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        if (more){
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    const ConvGeom& g = p.g;
    #pragma unroll
    for (int i = 0; i < 4; ++i){
        int m = m0 + ty + 16 * i;
        if (m >= p.gm) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j){
            int n = n0 + tx + 16 * j;
            if (n >= p.gn) continue;
            if (MODE == FWD){
                out[(size_t)m * g.k + group * p.kg + n] = FromF<OutT>(acc[i][j]);
            }
            else if (MODE == DGRAD){
                out[(size_t)m * g.c + group * p.cg + n] = FromF<OutT>(acc[i][j]);
            }
            else{
                atomicAdd((float*)out + (size_t)(group * p.kg + m) * p.gn + n, acc[i][j]);
            }
        }
    }
}

ConvArgs MakeArgs(const ConvGeom& g, int mode){
    ConvArgs p;
    p.g = g;
    p.cg = g.c / g.groups;
    p.kg = g.k / g.groups;
    if (mode == FWD){ p.gm = g.n * g.p * g.q; p.gn = p.kg; p.gk = g.r * g.s * p.cg; }
    else if (mode == DGRAD){ p.gm = g.n * g.h * g.w; p.gn = p.cg; p.gk = g.r * g.s * p.kg; }
    else { p.gm = p.kg; p.gn = g.r * g.s * p.cg; p.gk = g.n * g.p * g.q; }
    p.splits = 1;
    p.k_per_split = p.gk;
    if (mode == WGRAD){
        // 输出很小、Kdim 很长：切成若干段并行，目标约 4 个 block / SM
        int tiles = ((p.gm + SBM - 1) / SBM) * ((p.gn + SBN - 1) / SBN) * g.groups;
        int want = std::max(1, 4 * DeviceSmCount() / tiles);
        int max_splits = std::max(1, p.gk / (SBK * 16));
        p.splits = std::min(want, max_splits);
        p.k_per_split = ((p.gk + p.splits - 1) / p.splits + SBK - 1) / SBK * SBK;
        p.splits = (p.gk + p.k_per_split - 1) / p.k_per_split;
    }
    return p;
}

template <typename T, typename OutT, int MODE, bool AK, bool BK>
void Launch(const T* a, const T* b, OutT* out, const ConvArgs& p){
    dim3 grid((p.gm + SBM - 1) / SBM, (p.gn + SBN - 1) / SBN, p.g.groups * p.splits);
    NNCheck(grid.y <= 65535 && grid.z <= 65535, "conv: grid too large");
    ConvGemmSimt<T, OutT, MODE, AK, BK><<<grid, 256>>>(a, b, out, p);
}

}  // namespace

template <typename T>
void ConvForward(const TinyTensor<T>& x, const TinyTensor<T>& w, TinyTensor<T>& y,
                 int pad_h, int pad_w, int stride_h, int stride_w, int groups){
    NNCheckGpu(x, "conv_forward"); NNCheckGpu(w, "conv_forward");
    ConvGeom g = MakeConvGeom(x.shape, w.shape, pad_h, pad_w, stride_h, stride_w, groups);
    y.resize_discard({g.n, g.p, g.q, g.k});
    ConvForwardImpl(x.p_data, w.p_data, y.p_data, g);
}

template <typename T>
void ConvForwardImpl(const T* x, const T* w, T* y, const ConvGeom& g){
    if (ConvStemSupported(g)){
        ConvStem<T>(FWD, x, w, y, g);
        return;
    }
    if (ConvGroupedSupported(g)){
        ConvGrouped<T>(FWD, x, w, y, g);
        return;
    }
    if (UseF32<T>(g, FWD)){
        ConvF32(FWD, (const float*)x, (const float*)w, (float*)y, g);
        return;
    }
    if (UseTensorCore<T>(g, FWD)){
        ConvTensorCore(FWD, (const bf16*)x, (const bf16*)w, y, g);
        return;
    }
    ConvArgs p = MakeArgs(g, FWD);
    Launch<T, T, FWD, true, true>(x, w, y, p);
}

template <typename T>
void ConvBackwardData(const TinyTensor<T>& dy, const TinyTensor<T>& w, TinyTensor<T>& dx, const std::vector<int>& x_shape,
                      int pad_h, int pad_w, int stride_h, int stride_w, int groups){
    NNCheckGpu(dy, "conv_backward_data"); NNCheckGpu(w, "conv_backward_data");
    ConvGeom g = MakeConvGeom(x_shape, w.shape, pad_h, pad_w, stride_h, stride_w, groups);
    NNCheck(dy.shape == std::vector<int>({g.n, g.p, g.q, g.k}), "conv_backward_data: dy shape");
    dx.resize_discard(x_shape);
    if (ConvGroupedSupported(g)){
        ConvGrouped<T>(DGRAD, dy.p_data, w.p_data, dx.p_data, g);
        return;
    }
    if (UseF32<T>(g, DGRAD)){
        ConvF32(DGRAD, (const float*)dy.p_data, (const float*)w.p_data, (float*)dx.p_data, g);
        return;
    }
    if (UseTensorCore<T>(g, DGRAD)){
        ConvTensorCore(DGRAD, (const bf16*)dy.p_data, (const bf16*)w.p_data, dx.p_data, g);
        return;
    }
    ConvArgs p = MakeArgs(g, DGRAD);
    Launch<T, T, DGRAD, true, false>(dy.p_data, w.p_data, dx.p_data, p);
}

template <typename T>
void ConvBackwardWeight(const TinyTensor<T>& x, const TinyTensor<T>& dy, TinyTensor<float>& dw,
                        int pad_h, int pad_w, int stride_h, int stride_w, int groups, bool accumulate){
    NNCheckGpu(x, "conv_backward_weight"); NNCheckGpu(dy, "conv_backward_weight"); NNCheckGpu(dw, "conv_backward_weight");
    ConvGeom g = MakeConvGeom(x.shape, dw.shape, pad_h, pad_w, stride_h, stride_w, groups);
    NNCheck(dy.shape == std::vector<int>({g.n, g.p, g.q, g.k}), "conv_backward_weight: dy shape");
    if (!accumulate) dw.zeros();
    if (ConvStemSupported(g)){
        ConvStem<T>(WGRAD, dy.p_data, x.p_data, dw.p_data, g);
        return;
    }
    if (ConvGroupedSupported(g)){
        ConvGrouped<T>(WGRAD, dy.p_data, x.p_data, dw.p_data, g);
        return;
    }
    if (UseF32<T>(g, WGRAD)){
        ConvF32(WGRAD, (const float*)dy.p_data, (const float*)x.p_data, dw.p_data, g);
        return;
    }
    if (UseTensorCore<T>(g, WGRAD)){
        ConvTensorCore(WGRAD, (const bf16*)dy.p_data, (const bf16*)x.p_data, dw.p_data, g);
        return;
    }
    ConvArgs p = MakeArgs(g, WGRAD);
    Launch<T, float, WGRAD, false, false>(dy.p_data, x.p_data, dw.p_data, p);
}

#define INSTANTIATE(T) \
    template void ConvForward<T>(const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<T>&, int, int, int, int, int); \
    template void ConvForwardImpl<T>(const T*, const T*, T*, const ConvGeom&); \
    template void ConvBackwardData<T>(const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<T>&, const std::vector<int>&, \
                                      int, int, int, int, int); \
    template void ConvBackwardWeight<T>(const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<float>&, int, int, int, int, \
                                        int, bool);
INSTANTIATE(float)
INSTANTIATE(bf16)
