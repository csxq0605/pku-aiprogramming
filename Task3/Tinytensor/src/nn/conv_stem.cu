// 输入只有 3 个通道的第一层 3x3 卷积（步长 1）：直接卷积
// 这一层每个输出只有 27 次乘加，GEMM 的 K 维只有 27，tile 大部分空转；瓶颈是写 N*H*W*K 的输出。
// block 常驻（grid 约为 SM 数的若干倍），依次处理若干段“一行里的 64 个输出像素 x 64 个输出通道”：
// 输入的 3 行 x 66 列放进共享内存（通道补到 4 个，一个抽头一次 float4 读出），
// lane 负责相邻 2 个输出通道（54 个权重放在寄存器里，block 开始时经共享内存合并读入），warp 依次处理像素，
// 写回是连续的 128 / 256 字节。
// 权重梯度：同样的 tile，lane 累加自己 2 个通道的 54 个梯度，block 内先在共享内存里归约，再 atomicAdd 到全局。
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

constexpr int C = 3, R = 3, S = 3, RSC = R * S * C, TQ = 64, TW = TQ + S - 1, THREADS = 256;

// tile[r][j] = x[n][oh - pad + r][ow0 - pad + j][0..2]，第 4 个分量为 0
template <typename T>
__device__ __forceinline__ void LoadTile(float4 (*tile)[TW], const T* x, const ConvGeom& g, int n, int oh, int ow0){
    for (int idx = threadIdx.x; idx < R * TW; idx += THREADS){
        int r = idx / TW, j = idx % TW;
        int ih = oh - g.pad_h + r, iw = ow0 - g.pad_w + j;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ih >= 0 && ih < g.h && iw >= 0 && iw < g.w){
            const T* src = x + (((size_t)n * g.h + ih) * g.w + iw) * C;
            v.x = ToF(src[0]); v.y = ToF(src[1]); v.z = ToF(src[2]);
        }
        tile[r][j] = v;
    }
}

// 第 row 段（一行 (n, oh) 里从 ow0 开始的 64 个像素）
__device__ __forceinline__ void RowOf(int row, const ConvGeom& g, int& n, int& oh, int& ow0){
    const int segs = (g.q + TQ - 1) / TQ;
    ow0 = (row % segs) * TQ; int t = row / segs; oh = t % g.p; n = t / g.p;
}

__device__ __forceinline__ void Store2(float* p, float a, float b){ *(float2*)p = make_float2(a, b); }
__device__ __forceinline__ void Store2(bf16* p, float a, float b){ *(__nv_bfloat162*)p = __floats2bfloat162_rn(a, b); }
__device__ __forceinline__ void Load2(const float* p, float& a, float& b){ float2 t = *(const float2*)p; a = t.x; b = t.y; }
__device__ __forceinline__ void Load2(const bf16* p, float& a, float& b){
    float2 t = __bfloat1622float2(*(const __nv_bfloat162*)p); a = t.x; b = t.y;
}

// grid: (blocks, K / 64)；block 按步长 gridDim.x 遍历所有段
template <typename T>
__global__ void __launch_bounds__(THREADS) StemFwd(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y,
                                                   const ConvGeom g){
    __shared__ float4 tile[R][TW];
    __shared__ float ws[64 * RSC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k = blockIdx.y * 64 + lane * 2;
    for (int i = threadIdx.x; i < 64 * RSC; i += THREADS) ws[i] = ToF(w[(size_t)blockIdx.y * 64 * RSC + i]);
    __syncthreads();
    float w0[RSC], w1[RSC];
    #pragma unroll
    for (int i = 0; i < RSC; ++i){ w0[i] = ws[(lane * 2) * RSC + i]; w1[i] = ws[(lane * 2 + 1) * RSC + i]; }
    const int rows = g.n * g.p * ((g.q + TQ - 1) / TQ);
    for (int row = blockIdx.x; row < rows; row += gridDim.x){
        int n, oh, ow0;
        RowOf(row, g, n, oh, ow0);
        __syncthreads();
        LoadTile(tile, x, g, n, oh, ow0);
        __syncthreads();
        for (int j = warp; j < TQ && ow0 + j < g.q; j += THREADS / 32){
            float a0 = 0.f, a1 = 0.f;
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int s = 0; s < S; ++s){
                    float4 v = tile[r][j + s];
                    const int i = (r * S + s) * C;
                    a0 = fmaf(v.x, w0[i], a0); a0 = fmaf(v.y, w0[i + 1], a0); a0 = fmaf(v.z, w0[i + 2], a0);
                    a1 = fmaf(v.x, w1[i], a1); a1 = fmaf(v.y, w1[i + 1], a1); a1 = fmaf(v.z, w1[i + 2], a1);
                }
            Store2(y + (((size_t)n * g.p + oh) * g.q + ow0 + j) * g.k + k, a0, a1);
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(THREADS) StemWgrad(const T* __restrict__ dy, const T* __restrict__ x, float* __restrict__ dw,
                                                     const ConvGeom g){
    __shared__ float4 tile[R][TW];
    __shared__ float red[64 * RSC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int k = blockIdx.y * 64 + lane * 2;
    const int rows = g.n * g.p * ((g.q + TQ - 1) / TQ);
    float a0[RSC], a1[RSC];
    #pragma unroll
    for (int i = 0; i < RSC; ++i){ a0[i] = 0.f; a1[i] = 0.f; }
    for (int i = threadIdx.x; i < 64 * RSC; i += THREADS) red[i] = 0.f;
    for (int row = blockIdx.x; row < rows; row += gridDim.x){
        int n, oh, ow0;
        RowOf(row, g, n, oh, ow0);
        // 先把本 warp 这一段的 8 个像素的 dy 全部发出去，与 tile 的读入重叠
        constexpr int PW = TQ / (THREADS / 32);
        float dv[PW][2];
        #pragma unroll
        for (int i = 0; i < PW; ++i){
            const int j = warp + i * (THREADS / 32);
            dv[i][0] = dv[i][1] = 0.f;
            if (ow0 + j < g.q) Load2(dy + (((size_t)n * g.p + oh) * g.q + ow0 + j) * g.k + k, dv[i][0], dv[i][1]);
        }
        __syncthreads();
        LoadTile(tile, x, g, n, oh, ow0);
        __syncthreads();
        #pragma unroll
        for (int pi = 0; pi < PW; ++pi){
            const int j = warp + pi * (THREADS / 32);
            const float d0 = dv[pi][0], d1 = dv[pi][1];   // 越界像素的 dy 为 0，tile 也已补 0
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int s = 0; s < S; ++s){
                    float4 v = tile[r][j + s];
                    const int i = (r * S + s) * C;
                    a0[i] = fmaf(v.x, d0, a0[i]); a0[i + 1] = fmaf(v.y, d0, a0[i + 1]); a0[i + 2] = fmaf(v.z, d0, a0[i + 2]);
                    a1[i] = fmaf(v.x, d1, a1[i]); a1[i + 1] = fmaf(v.y, d1, a1[i + 1]); a1[i + 2] = fmaf(v.z, d1, a1[i + 2]);
                }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < RSC; ++i){
        atomicAdd(&red[(lane * 2) * RSC + i], a0[i]);
        atomicAdd(&red[(lane * 2 + 1) * RSC + i], a1[i]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * RSC; i += THREADS) atomicAdd(dw + (size_t)blockIdx.y * 64 * RSC + i, red[i]);
}

}  // namespace

bool ConvStemSupported(const ConvGeom& g){
    return g.c == C && g.r == R && g.s == S && g.stride_h == 1 && g.stride_w == 1 && g.groups == 1 && g.k % 64 == 0;
}

template <typename T>
void ConvStem(int mode, const T* a, const T* b, void* out, const ConvGeom& g){
    static int sms = 0;
    if (!sms){
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    // 每个 SM 若干个常驻 block；段数少时不超过段数
    const int rows = g.n * g.p * ((g.q + TQ - 1) / TQ);
    if (mode == 0) StemFwd<T><<<dim3(std::min(rows, sms * 8), g.k / 64), THREADS>>>(a, b, (T*)out, g);
    else StemWgrad<T><<<dim3(std::min(rows, sms * 4), g.k / 64), THREADS>>>(a, b, (float*)out, g);
}

template void ConvStem<float>(int, const float*, const float*, void*, const ConvGeom&);
template void ConvStem<bf16>(int, const bf16*, const bf16*, void*, const ConvGeom&);
