// BatchNorm（NHWC，按通道）与 LayerNorm（按最后一维）
#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

// ---------- BatchNorm ----------
// 均值 / 方差和仿射系数：前向和反向（重算 ReLU 掩码）共用同一套显式舍入的算式，保证两边逐位一致
__device__ __forceinline__ void BnMeanVar(const float* stats, int c, int C, int rows, float& mean, float& var){
    mean = stats[c] / rows;
    var = fmaxf(__fmaf_rn(-mean, mean, stats[C + c] / rows), 0.f);
}
__device__ __forceinline__ void BnAffine(float mean, float var, float gamma, float beta, float eps, float& sc, float& sf){
    sc = gamma * rsqrtf(var + eps);
    sf = __fmaf_rn(-mean, sc, beta);
}
// 按列（通道）求和：x 视为 [rows, C]。block = 32 列 x 8 行，grid.y 把行切成若干段，结果 atomicAdd
template <typename T, bool BACKWARD, bool RELU>
__global__ void BnReduce(const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ dy,
                         const float* __restrict__ stats, float* __restrict__ out, int rows, int C, float eps){
    __shared__ float s1[8][33], s2[8][33];
    const int c = blockIdx.x * 32 + threadIdx.x;
    float a = 0.f, b = 0.f;
    if (c < C){
        float mean = 0.f, rstd = 0.f;
        if (BACKWARD){
            mean = stats[c] / rows;
            rstd = rsqrtf(fmaxf(stats[C + c] / rows - mean * mean, 0.f) + eps);
        }
        for (int r = blockIdx.y * 8 + threadIdx.y; r < rows; r += gridDim.y * 8){
            size_t i = (size_t)r * C + c;
            float v = ToF(x[i]);
            if (!BACKWARD){
                a += v;
                b += v * v;
            }
            else{
                float g = ToF(dy[i]);
                if (RELU && ToF(y[i]) <= 0.f) g = 0.f;
                a += g;
                b += g * (v - mean) * rstd;
            }
        }
    }
    s1[threadIdx.y][threadIdx.x] = a;
    s2[threadIdx.y][threadIdx.x] = b;
    __syncthreads();
    if (threadIdx.y == 0 && c < C){
        #pragma unroll
        for (int k = 1; k < 8; ++k){ a += s1[k][threadIdx.x]; b += s2[k][threadIdx.x]; }
        atomicAdd(out + c, a);
        atomicAdd(out + C + c, b);
    }
}

// y = act(x * scale[c] + shift[c] [+ residual])；scale / shift 每个 block 开头算好放在共享内存
// TRAIN 时 block 0 顺带更新 running 统计量
template <typename T, bool TRAIN, bool RES, bool RELU>
__global__ void BnApply(const T* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                        const float* __restrict__ stats_or_mean, const float* __restrict__ var_eval,
                        float* __restrict__ running_mean, float* __restrict__ running_var,
                        const T* __restrict__ residual, T* __restrict__ y, size_t total, int rows, int C,
                        float momentum, float eps){
    extern __shared__ float sh[];
    float* scale = sh;
    float* shift = sh + C;
    for (int c = threadIdx.x; c < C; c += blockDim.x){
        float mean, var;
        if (TRAIN){
            BnMeanVar(stats_or_mean, c, C, rows, mean, var);
            if (blockIdx.x == 0){
                running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1.f - momentum) * running_var[c] + momentum * var * rows / fmaxf(rows - 1, 1);
            }
        }
        else{
            mean = stats_or_mean[c];
            var = var_eval[c];
        }
        BnAffine(mean, var, gamma[c], beta[c], eps, scale[c], shift[c]);
    }
    __syncthreads();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % C;
        float v = __fmaf_rn(ToF(x[i]), scale[c], shift[c]);
        if (RES) v += ToF(residual[i]);
        if (RELU) v = fmaxf(v, 0.f);
        y[i] = FromF<T>(v);
    }
}

// dx = gamma * rstd * (g - sum_g / M - xhat * sum_gx / M)，g = dy * relu'；block 0 把 sum_g / sum_gx 累加到 dbeta / dgamma
template <typename T, bool RELU, bool DRES>
__global__ void BnBackwardApply(const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ dy,
                                const float* __restrict__ gamma, const float* __restrict__ stats,
                                const float* __restrict__ red, float* __restrict__ dgamma, float* __restrict__ dbeta,
                                T* __restrict__ dx, T* __restrict__ dres, size_t total, int rows, int C, float eps){
    extern __shared__ float sh[];
    float* k_g = sh;           // gamma * rstd
    float* mean_s = sh + C;
    float* rstd_s = sh + 2 * C;
    float* mg = sh + 3 * C;    // sum_g / M
    float* mgx = sh + 4 * C;   // sum_gx / M
    for (int c = threadIdx.x; c < C; c += blockDim.x){
        float mean = stats[c] / rows;
        float rstd = rsqrtf(fmaxf(stats[C + c] / rows - mean * mean, 0.f) + eps);
        k_g[c] = gamma[c] * rstd;
        mean_s[c] = mean;
        rstd_s[c] = rstd;
        mg[c] = red[c] / rows;
        mgx[c] = red[C + c] / rows;
        if (blockIdx.x == 0){
            dbeta[c] += red[c];
            dgamma[c] += red[C + c];
        }
    }
    __syncthreads();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % C;
        float g = ToF(dy[i]);
        if (RELU && ToF(y[i]) <= 0.f) g = 0.f;
        if (DRES) dres[i] = FromF<T>(g);
        float xhat = (ToF(x[i]) - mean_s[c]) * rstd_s[c];
        dx[i] = FromF<T>(k_g[c] * (g - mg[c] - xhat * mgx[c]));
    }
}

// ---------- BatchNorm 的向量化版本：每个线程一次处理同一像素的 8 个连续通道（bf16 一次 16 字节，fp32 两次）----------
// 上面的标量版本每次只读 2 / 4 字节，BnApply 还要对每个元素做 64 位取模，A100 上只能跑到带宽的 1/4 左右。
// 条件：C % 8 == 0 且 C / 8 整除 256（一个 block 每轮处理 256 / (C / 8) 个像素）
constexpr int BN_VEC = 8, BN_THREADS = 256;

__device__ __forceinline__ void Load8(const float* p, float* v){
    float4 a = __ldg((const float4*)p), b = __ldg((const float4*)p + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void Load8(const bf16* p, float* v){
    uint4 t = __ldg((const uint4*)p);
    const __nv_bfloat162* h = (const __nv_bfloat162*)&t;
    #pragma unroll
    for (int j = 0; j < 4; ++j){ float2 f = __bfloat1622float2(h[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
}
__device__ __forceinline__ void Store8(float* p, const float* v){
    ((float4*)p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    ((float4*)p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void Store8(bf16* p, const float* v){
    uint4 t;
    __nv_bfloat162* h = (__nv_bfloat162*)&t;
    #pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
    *(uint4*)p = t;
}

// ReLU 掩码 y > 0：没有残差时 y = T(max(x * sc + sf, 0))，可以由 x 精确重算（与前向同一算式、同样舍入），
// 反向就不用再读一遍 y（BN 反向的两个 kernel 各省一份激活的读取）
template <typename T>
__device__ __forceinline__ bool BnReluOn(float x, float sc, float sf){
    return ToF(FromF<T>(fmaxf(__fmaf_rn(x, sc, sf), 0.f))) > 0.f;
}

// 按通道求和（同 BnReduce）：线程 (r0, cv) 负责通道 cv*8 .. cv*8+7，每轮跨 RP 个像素；block 内经共享内存归约后 atomicAdd
// RECOMP：ReLU 掩码由 x 重算（需要 gamma / beta），不读 y
template <typename T, bool BACKWARD, bool RELU, bool RECOMP = false>
__global__ void __launch_bounds__(BN_THREADS) BnReduceVec(const T* __restrict__ x, const T* __restrict__ y,
        const T* __restrict__ dy, const float* __restrict__ stats, float* __restrict__ out, int rows, int C, float eps,
        const float* __restrict__ gamma = nullptr, const float* __restrict__ beta = nullptr){
    __shared__ float sa[BN_THREADS * BN_VEC], sb[BN_THREADS * BN_VEC];   // [RP][C]
    const int CV = C / BN_VEC, RP = BN_THREADS / CV;
    const int cv = threadIdx.x % CV, r0 = threadIdx.x / CV, c0 = cv * BN_VEC;
    float a[BN_VEC] = {}, b[BN_VEC] = {}, mean[BN_VEC], rstd[BN_VEC], sc[BN_VEC], sf[BN_VEC];
    if (BACKWARD){
        #pragma unroll
        for (int j = 0; j < BN_VEC; ++j){
            float var;
            BnMeanVar(stats, c0 + j, C, rows, mean[j], var);
            rstd[j] = rsqrtf(var + eps);
            if (RECOMP) BnAffine(mean[j], var, gamma[c0 + j], beta[c0 + j], eps, sc[j], sf[j]);
        }
    }
    for (int r = blockIdx.x * RP + r0; r < rows; r += gridDim.x * RP){
        const size_t i = (size_t)r * C + c0;
        float v[BN_VEC];
        Load8(x + i, v);
        if (!BACKWARD){
            #pragma unroll
            for (int j = 0; j < BN_VEC; ++j){ a[j] += v[j]; b[j] += v[j] * v[j]; }
        }
        else{
            float g[BN_VEC], yy[BN_VEC];
            Load8(dy + i, g);
            if (RELU && !RECOMP) Load8(y + i, yy);
            #pragma unroll
            for (int j = 0; j < BN_VEC; ++j){
                if (RELU && (RECOMP ? !BnReluOn<T>(v[j], sc[j], sf[j]) : yy[j] <= 0.f)) g[j] = 0.f;
                a[j] += g[j];
                b[j] += g[j] * (v[j] - mean[j]) * rstd[j];
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < BN_VEC; ++j){ sa[r0 * C + c0 + j] = a[j]; sb[r0 * C + c0 + j] = b[j]; }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += BN_THREADS){
        float s1 = 0.f, s2 = 0.f;
        for (int r = 0; r < RP; ++r){ s1 += sa[r * C + c]; s2 += sb[r * C + c]; }
        atomicAdd(out + c, s1);
        atomicAdd(out + C + c, s2);
    }
}

// 同 BnApply，按 8 个通道一组处理；grid-stride 时通道组号增量更新，不做 64 位取模
template <typename T, bool TRAIN, bool RES, bool RELU>
__global__ void __launch_bounds__(BN_THREADS) BnApplyVec(const T* __restrict__ x, const float* __restrict__ gamma,
        const float* __restrict__ beta, const float* __restrict__ stats_or_mean, const float* __restrict__ var_eval,
        float* __restrict__ running_mean, float* __restrict__ running_var, const T* __restrict__ residual,
        T* __restrict__ y, size_t total, int rows, int C, float momentum, float eps){
    extern __shared__ float sh[];
    float* scale = sh;
    float* shift = sh + C;
    for (int c = threadIdx.x; c < C; c += blockDim.x){
        float mean, var;
        if (TRAIN){
            BnMeanVar(stats_or_mean, c, C, rows, mean, var);
            if (blockIdx.x == 0){
                running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1.f - momentum) * running_var[c] + momentum * var * rows / fmaxf(rows - 1, 1);
            }
        }
        else{
            mean = stats_or_mean[c];
            var = var_eval[c];
        }
        BnAffine(mean, var, gamma[c], beta[c], eps, scale[c], shift[c]);
    }
    __syncthreads();
    const int CV = C / BN_VEC;
    const size_t nv = total / BN_VEC, stride = (size_t)gridDim.x * blockDim.x;
    size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    int cv = (int)(v % CV);
    const int step = (int)(stride % CV);
    for (; v < nv; v += stride){
        float xv[BN_VEC], rv[BN_VEC];
        Load8(x + v * BN_VEC, xv);
        if (RES) Load8(residual + v * BN_VEC, rv);
        const float* sc = scale + cv * BN_VEC;
        const float* sf = shift + cv * BN_VEC;
        #pragma unroll
        for (int j = 0; j < BN_VEC; ++j){
            float o = __fmaf_rn(xv[j], sc[j], sf[j]);
            if (RES) o += rv[j];
            if (RELU) o = fmaxf(o, 0.f);
            xv[j] = o;
        }
        Store8(y + v * BN_VEC, xv);
        cv += step;
        if (cv >= CV) cv -= CV;
    }
}

// 同 BnBackwardApply，按 8 个通道一组处理；RECOMP 同 BnReduceVec（ReLU 掩码由 x 重算，不读 y）
template <typename T, bool RELU, bool DRES, bool RECOMP = false>
__global__ void __launch_bounds__(BN_THREADS) BnBackwardApplyVec(const T* __restrict__ x, const T* __restrict__ y,
        const T* __restrict__ dy, const float* __restrict__ gamma, const float* __restrict__ stats,
        const float* __restrict__ red, float* __restrict__ dgamma, float* __restrict__ dbeta,
        T* __restrict__ dx, T* __restrict__ dres, size_t total, int rows, int C, float eps,
        const float* __restrict__ beta = nullptr){
    extern __shared__ float sh[];
    float* k_g = sh;           // gamma * rstd（= 前向的 scale）
    float* mean_s = sh + C;
    float* rstd_s = sh + 2 * C;
    float* mg = sh + 3 * C;    // sum_g / M
    float* mgx = sh + 4 * C;   // sum_gx / M
    float* sf_s = sh + 5 * C;  // 前向的 shift（RECOMP）
    for (int c = threadIdx.x; c < C; c += blockDim.x){
        float mean, var;
        BnMeanVar(stats, c, C, rows, mean, var);
        float rstd = rsqrtf(var + eps);
        if (RECOMP) BnAffine(mean, var, gamma[c], beta[c], eps, k_g[c], sf_s[c]);
        else k_g[c] = gamma[c] * rstd;
        mean_s[c] = mean;
        rstd_s[c] = rstd;
        mg[c] = red[c] / rows;
        mgx[c] = red[C + c] / rows;
        if (blockIdx.x == 0){
            dbeta[c] += red[c];
            dgamma[c] += red[C + c];
        }
    }
    __syncthreads();
    const int CV = C / BN_VEC;
    const size_t nv = total / BN_VEC, stride = (size_t)gridDim.x * blockDim.x;
    size_t v = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    int cv = (int)(v % CV);
    const int step = (int)(stride % CV);
    for (; v < nv; v += stride){
        float g[BN_VEC], yy[BN_VEC], xv[BN_VEC];
        Load8(dy + v * BN_VEC, g);
        if (RELU && !RECOMP) Load8(y + v * BN_VEC, yy);
        Load8(x + v * BN_VEC, xv);
        const int c0 = cv * BN_VEC;
        #pragma unroll
        for (int j = 0; j < BN_VEC; ++j){
            if (RELU && (RECOMP ? !BnReluOn<T>(xv[j], k_g[c0 + j], sf_s[c0 + j]) : yy[j] <= 0.f)) g[j] = 0.f;
            const float xhat = (xv[j] - mean_s[c0 + j]) * rstd_s[c0 + j];
            xv[j] = k_g[c0 + j] * (g[j] - mg[c0 + j] - xhat * mgx[c0 + j]);
        }
        if (DRES) Store8(dres + v * BN_VEC, g);
        Store8(dx + v * BN_VEC, xv);
        cv += step;
        if (cv >= CV) cv -= CV;
    }
}

bool BnVecOk(int C, std::initializer_list<const void*> ptrs){
    if (C % BN_VEC != 0 || C / BN_VEC > BN_THREADS || BN_THREADS % (C / BN_VEC) != 0) return false;
    for (const void* p : ptrs) if ((size_t)p % 16 != 0) return false;
    return true;
}

// 向量化 reduce 的 block 数：每个线程至少处理约 8 轮，最多每个 SM 4 个 block
int BnReduceBlocks(int C, int rows){
    const int rp = BN_THREADS / (C / BN_VEC);
    return std::max(1, std::min((rows / rp + 7) / 8, DeviceSmCount() * 4));
}
int BnApplyBlocks(size_t total){
    return (int)std::max<size_t>(1, std::min<size_t>((total / BN_VEC + BN_THREADS - 1) / BN_THREADS, DeviceSmCount() * 8));
}

// TT_BN_RECOMPUTE=0：反向照旧读 y 求 ReLU 掩码（对照实验用）
bool RecomputeMask(){
    static const bool on = [](){ const char* e = getenv("TT_BN_RECOMPUTE"); return !(e && atoi(e) == 0); }();
    return on;
}

int ReduceSplits(int C, int rows){
    int col_blocks = (C + 31) / 32;
    int splits = std::max(1, 192 / col_blocks);
    return std::min(splits, std::max(1, rows / 64));
}

int ChannelsOf(const std::vector<int>& shape){
    NNCheck(!shape.empty(), "norm: empty shape");
    return shape.back();
}

// ---------- LayerNorm：一个 warp 处理一行 ----------
template <typename T>
__global__ void LnForward(const T* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                          T* __restrict__ y, float* __restrict__ mean_rstd, int rows, int D, float eps){
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= rows) return;
    const T* xr = x + (size_t)warp * D;
    float s = 0.f;
    for (int i = lane; i < D; i += 32) s += ToF(xr[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    const float mean = s / D;
    float v = 0.f;
    for (int i = lane; i < D; i += 32){ float d = ToF(xr[i]) - mean; v += d * d; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const float rstd = rsqrtf(v / D + eps);
    T* yr = y + (size_t)warp * D;
    for (int i = lane; i < D; i += 32) yr[i] = FromF<T>((ToF(xr[i]) - mean) * rstd * gamma[i] + beta[i]);
    if (lane == 0){ mean_rstd[2 * warp] = mean; mean_rstd[2 * warp + 1] = rstd; }
}

// 每个 warp 处理若干行；dgamma / dbeta 的部分和先放在共享内存里，block 结束时 atomicAdd
template <typename T>
__global__ void LnBackward(const T* __restrict__ x, const T* __restrict__ dy, const float* __restrict__ gamma,
                           const float* __restrict__ mean_rstd, T* __restrict__ dx, float* __restrict__ dgamma,
                           float* __restrict__ dbeta, int rows, int D){
    extern __shared__ float sh[];
    float* sg = sh;        // dgamma 部分和
    float* sb = sh + D;    // dbeta 部分和
    for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) sh[i] = 0.f;
    __syncthreads();
    const int lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int r = blockIdx.x * warps + threadIdx.x / 32; r < rows; r += gridDim.x * warps){
        const T* xr = x + (size_t)r * D;
        const T* dyr = dy + (size_t)r * D;
        const float mean = mean_rstd[2 * r], rstd = mean_rstd[2 * r + 1];
        float a = 0.f, b = 0.f;
        for (int i = lane; i < D; i += 32){
            float xh = (ToF(xr[i]) - mean) * rstd;
            float g = ToF(dyr[i]);
            float dxh = g * gamma[i];
            a += dxh;
            b += dxh * xh;
            atomicAdd(sg + i, g * xh);
            atomicAdd(sb + i, g);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1){
            a += __shfl_xor_sync(0xffffffff, a, o);
            b += __shfl_xor_sync(0xffffffff, b, o);
        }
        a /= D; b /= D;
        T* dxr = dx + (size_t)r * D;
        for (int i = lane; i < D; i += 32){
            float xh = (ToF(xr[i]) - mean) * rstd;
            float dxh = ToF(dyr[i]) * gamma[i];
            dxr[i] = FromF<T>(rstd * (dxh - a - xh * b));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x){
        atomicAdd(dgamma + i, sg[i]);
        atomicAdd(dbeta + i, sb[i]);
    }
}

}  // namespace

template <typename T>
void BatchNormTrain(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                    TinyTensor<float>& running_mean, TinyTensor<float>& running_var, TinyTensor<float>& stats,
                    TinyTensor<T>& y, const TinyTensor<T>* residual, bool relu, float momentum, float eps){
    NNCheckGpu(x, "batchnorm");
    const int C = ChannelsOf(x.shape);
    const int rows = (int)(x.size / C);
    NNCheck(gamma.size == (size_t)C && beta.size == (size_t)C, "batchnorm: gamma/beta size");
    NNCheck(!residual || residual->shape == x.shape, "batchnorm: residual shape");
    stats.resize_discard({2 * C});
    stats.zeros();
    y.resize_discard(x.shape);
    const size_t smem = 2 * C * sizeof(float);
    const T* res = residual ? residual->p_data : nullptr;
    if (BnVecOk(C, {x.p_data, y.p_data, res})){
        BnReduceVec<T, false, false><<<BnReduceBlocks(C, rows), BN_THREADS>>>(x.p_data, nullptr, nullptr, nullptr,
                                                                              stats.p_data, rows, C, eps);
        const int nb = BnApplyBlocks(x.size);
#define APPLY(R, A) BnApplyVec<T, true, R, A><<<nb, BN_THREADS, smem>>>(x.p_data, gamma.p_data, beta.p_data, stats.p_data,         nullptr, running_mean.p_data, running_var.p_data, res, y.p_data, x.size, rows, C, momentum, eps)
        if (res){ if (relu) APPLY(true, true); else APPLY(true, false); }
        else    { if (relu) APPLY(false, true); else APPLY(false, false); }
#undef APPLY
        return;
    }
    dim3 grid((C + 31) / 32, ReduceSplits(C, rows));
    BnReduce<T, false, false><<<grid, dim3(32, 8)>>>(x.p_data, nullptr, nullptr, nullptr, stats.p_data, rows, C, eps);
    const int blocks = GridFor(x.size, 256);
    const int capped = std::min(blocks, 24 * 16);
#define APPLY(R, A) BnApply<T, true, R, A><<<capped, 256, smem>>>(x.p_data, gamma.p_data, beta.p_data, stats.p_data, \
        nullptr, running_mean.p_data, running_var.p_data, res, y.p_data, x.size, rows, C, momentum, eps)
    if (res){ if (relu) APPLY(true, true); else APPLY(true, false); }
    else    { if (relu) APPLY(false, true); else APPLY(false, false); }
#undef APPLY
}

template <typename T>
void BatchNormEval(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                   const TinyTensor<float>& running_mean, const TinyTensor<float>& running_var,
                   TinyTensor<T>& y, const TinyTensor<T>* residual, bool relu, float eps){
    NNCheckGpu(x, "batchnorm_eval");
    const int C = ChannelsOf(x.shape);
    const int rows = (int)(x.size / C);
    y.resize_discard(x.shape);
    const size_t smem = 2 * C * sizeof(float);
    const T* res = residual ? residual->p_data : nullptr;
    if (BnVecOk(C, {x.p_data, y.p_data, res})){
        const int nb = BnApplyBlocks(x.size);
#define APPLY(R, A) BnApplyVec<T, false, R, A><<<nb, BN_THREADS, smem>>>(x.p_data, gamma.p_data, beta.p_data,         running_mean.p_data, running_var.p_data, nullptr, nullptr, res, y.p_data, x.size, rows, C, 0.f, eps)
        if (res){ if (relu) APPLY(true, true); else APPLY(true, false); }
        else    { if (relu) APPLY(false, true); else APPLY(false, false); }
#undef APPLY
        return;
    }
    const int capped = std::min(GridFor(x.size, 256), 24 * 16);
#define APPLY(R, A) BnApply<T, false, R, A><<<capped, 256, smem>>>(x.p_data, gamma.p_data, beta.p_data, \
        running_mean.p_data, running_var.p_data, nullptr, nullptr, res, y.p_data, x.size, rows, C, 0.f, eps)
    if (res){ if (relu) APPLY(true, true); else APPLY(true, false); }
    else    { if (relu) APPLY(false, true); else APPLY(false, false); }
#undef APPLY
}

template <typename T>
void BatchNormBackward(const TinyTensor<T>& x, const TinyTensor<T>& y, const TinyTensor<T>& dy,
                       const TinyTensor<float>& gamma, const TinyTensor<float>& beta, const TinyTensor<float>& stats,
                       TinyTensor<T>& dx, TinyTensor<float>& dgamma, TinyTensor<float>& dbeta,
                       TinyTensor<T>* dresidual, bool relu, float eps){
    NNCheckGpu(x, "batchnorm_backward");
    const int C = ChannelsOf(x.shape);
    const int rows = (int)(x.size / C);
    NNCheck(dy.shape == x.shape && y.shape == x.shape, "batchnorm_backward: shape");
    TinyTensor<float> red(std::vector<int>{2 * C}, "gpu");
    red.zeros();
    dx.resize_discard(x.shape);
    T* dres = nullptr;
    if (dresidual){
        dresidual->resize_discard(x.shape);
        dres = dresidual->p_data;
    }
    const size_t smem = 6 * C * sizeof(float);
    if (BnVecOk(C, {x.p_data, y.p_data, dy.p_data, dx.p_data, dres})){
        const int rb = BnReduceBlocks(C, rows), nb = BnApplyBlocks(x.size);
        // 没有残差的 BN+ReLU（VGG 全部、ResNet 每块的第一个 BN）：掩码由 x 重算，两个 kernel 都不读 y
        if (relu && !dres && RecomputeMask()){
            BnReduceVec<T, true, true, true><<<rb, BN_THREADS>>>(x.p_data, nullptr, dy.p_data, stats.p_data, red.p_data,
                                                                 rows, C, eps, gamma.p_data, beta.p_data);
            BnBackwardApplyVec<T, true, false, true><<<nb, BN_THREADS, smem>>>(x.p_data, nullptr, dy.p_data, gamma.p_data,
                stats.p_data, red.p_data, dgamma.p_data, dbeta.p_data, dx.p_data, nullptr, x.size, rows, C, eps, beta.p_data);
            return;
        }
        if (relu) BnReduceVec<T, true, true><<<rb, BN_THREADS>>>(x.p_data, y.p_data, dy.p_data, stats.p_data, red.p_data, rows, C, eps);
        else      BnReduceVec<T, true, false><<<rb, BN_THREADS>>>(x.p_data, y.p_data, dy.p_data, stats.p_data, red.p_data, rows, C, eps);
#define APPLY(A, D) BnBackwardApplyVec<T, A, D><<<nb, BN_THREADS, smem>>>(x.p_data, y.p_data, dy.p_data, gamma.p_data,         stats.p_data, red.p_data, dgamma.p_data, dbeta.p_data, dx.p_data, dres, x.size, rows, C, eps)
        if (dres){ if (relu) APPLY(true, true); else APPLY(false, true); }
        else     { if (relu) APPLY(true, false); else APPLY(false, false); }
#undef APPLY
        return;
    }
    dim3 grid((C + 31) / 32, ReduceSplits(C, rows));
    if (relu) BnReduce<T, true, true><<<grid, dim3(32, 8)>>>(x.p_data, y.p_data, dy.p_data, stats.p_data, red.p_data, rows, C, eps);
    else      BnReduce<T, true, false><<<grid, dim3(32, 8)>>>(x.p_data, y.p_data, dy.p_data, stats.p_data, red.p_data, rows, C, eps);
    const int capped = std::min(GridFor(x.size, 256), 24 * 16);
#define APPLY(A, D) BnBackwardApply<T, A, D><<<capped, 256, smem>>>(x.p_data, y.p_data, dy.p_data, gamma.p_data, \
        stats.p_data, red.p_data, dgamma.p_data, dbeta.p_data, dx.p_data, dres, x.size, rows, C, eps)
    if (dres){ if (relu) APPLY(true, true); else APPLY(false, true); }
    else     { if (relu) APPLY(true, false); else APPLY(false, false); }
#undef APPLY
}

template <typename T>
void LayerNormForward(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                      TinyTensor<T>& y, TinyTensor<float>& mean_rstd, float eps){
    NNCheckGpu(x, "layernorm");
    const int D = ChannelsOf(x.shape);
    const int rows = (int)(x.size / D);
    NNCheck(gamma.size == (size_t)D && beta.size == (size_t)D, "layernorm: gamma/beta size");
    y.resize_discard(x.shape);
    mean_rstd.resize_discard({rows, 2});
    LnForward<T><<<(rows + 7) / 8, 256>>>(x.p_data, gamma.p_data, beta.p_data, y.p_data, mean_rstd.p_data, rows, D, eps);
}

template <typename T>
void LayerNormBackward(const TinyTensor<T>& x, const TinyTensor<T>& dy, const TinyTensor<float>& gamma,
                       const TinyTensor<float>& mean_rstd, TinyTensor<T>& dx,
                       TinyTensor<float>& dgamma, TinyTensor<float>& dbeta){
    NNCheckGpu(x, "layernorm_backward");
    const int D = ChannelsOf(x.shape);
    const int rows = (int)(x.size / D);
    dx.resize_discard(x.shape);
    const int blocks = std::min((rows + 7) / 8, 96);
    LnBackward<T><<<blocks, 256, 2 * D * sizeof(float)>>>(x.p_data, dy.p_data, gamma.p_data, mean_rstd.p_data,
                                                          dx.p_data, dgamma.p_data, dbeta.p_data, rows, D);
}

#define INSTANTIATE(T) \
    template void BatchNormTrain<T>(const TinyTensor<T>&, const TinyTensor<float>&, const TinyTensor<float>&, \
        TinyTensor<float>&, TinyTensor<float>&, TinyTensor<float>&, TinyTensor<T>&, const TinyTensor<T>*, bool, float, float); \
    template void BatchNormEval<T>(const TinyTensor<T>&, const TinyTensor<float>&, const TinyTensor<float>&, \
        const TinyTensor<float>&, const TinyTensor<float>&, TinyTensor<T>&, const TinyTensor<T>*, bool, float); \
    template void BatchNormBackward<T>(const TinyTensor<T>&, const TinyTensor<T>&, const TinyTensor<T>&, \
        const TinyTensor<float>&, const TinyTensor<float>&, const TinyTensor<float>&, TinyTensor<T>&, TinyTensor<float>&, TinyTensor<float>&, \
        TinyTensor<T>*, bool, float); \
    template void LayerNormForward<T>(const TinyTensor<T>&, const TinyTensor<float>&, const TinyTensor<float>&, \
        TinyTensor<T>&, TinyTensor<float>&, float); \
    template void LayerNormBackward<T>(const TinyTensor<T>&, const TinyTensor<T>&, const TinyTensor<float>&, \
        const TinyTensor<float>&, TinyTensor<T>&, TinyTensor<float>&, TinyTensor<float>&);
INSTANTIATE(float)
INSTANTIATE(bf16)
