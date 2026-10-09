// 多头自注意力（ViT，序列较短）：一个 block 负责一个 (batch, head)，Q/K/V 与注意力矩阵都放在共享内存里
// qkv: [B, L, 3, H, hd]（qkv 全连接层的输出直接 reshape），out: [B, L, H, hd]，probs: [B, H, L, L] fp32（反向用）
// 反向不保存 dP：r_i = sum_j P_ij dP_ij = dot(dO_i, O_i)，O_i 由 P 和 V 重算，于是 dS_ij = P_ij (dP_ij - r_i) 可以原地算
#include <cmath>
#include <type_traits>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

// 共享内存里 Q / K / V / dO 按输入类型 T 存（bf16 时占用减半，每个 SM 能放 2 个 block）；
// 行距 ld = hd + 4 / sizeof(T) 个元素，使行距是奇数个 32 位字，按列读 K / V（转置访问）时没有 bank 冲突
// 一个 warp 读一行，每个 lane 一次读相邻 2 个元素（hd 为偶数）；不做整数除法
template <typename T> struct Pair;
template <> struct Pair<float>{ using type = float2; };
template <> struct Pair<bf16>{ using type = __nv_bfloat162; };
template <typename T>
__device__ __forceinline__ void LoadRows(const T* __restrict__ src, size_t row_stride, T* dst, int L, int hd, int ld){
    using P2 = typename Pair<T>::type;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int l = warp; l < L; l += blockDim.x >> 5)
        for (int d = lane * 2; d < hd; d += 64){
            P2 v = *(const P2*)(src + l * row_stride + d);
            dst[l * ld + d] = v.x; dst[l * ld + d + 1] = v.y;
        }
}
template <typename T>
__device__ __forceinline__ void LoadHead(const T* __restrict__ qkv, T* dst, int b, int which, int h,
                                         int L, int H, int hd, int ld){
    LoadRows(qkv + ((size_t)b * L * 3 + which) * H * hd + (size_t)h * hd, (size_t)3 * H * hd, dst, L, hd, ld);
}
template <typename T> __host__ __device__ constexpr int RowPad(){ return 4 / (int)sizeof(T); }

// block 内的小矩阵乘（共享内存里的 fp32 矩阵）：C(i, j) = sum_k A(i, k) B(k, j)，A(i, k) = A[i*ars + k*acs]，B 同理
// 256 个线程排成 16 x 16，线程 (ty, tx) 负责 i = ty + 16a (a < TM)、j = tx + 16b (b < TN) 的 TM x TN 个元素：
// 每个 k 读 TM + TN 个数做 TM * TN 次乘加（原来每次乘加要读 2 个数）。
// 同一 warp 只有 2 个不同的 ty（A 的读取是广播），tx 连续（B 的读取落在不同 bank）。
// 越界的行 / 列读第 M-1 行 / 第 N-1 列（结果不写回），避免分支。epi(i, j, v) 处理每个有效元素。
template <int TM, int TN, typename TA, typename TB, typename Epi>
__device__ __forceinline__ void BlockGemm(const TA* A, int ars, int acs, const TB* B, int brs, int bcs,
                                          int M, int N, int Kd, Epi epi){
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    for (int i0 = 0; i0 < M; i0 += 16 * TM){
        for (int j0 = 0; j0 < N; j0 += 16 * TN){
            float acc[TM][TN];
            int ai[TM], bj[TN];
            #pragma unroll
            for (int a = 0; a < TM; ++a){
                ai[a] = min(i0 + ty + 16 * a, M - 1) * ars;
                #pragma unroll
                for (int b = 0; b < TN; ++b) acc[a][b] = 0.f;
            }
            #pragma unroll
            for (int b = 0; b < TN; ++b) bj[b] = min(j0 + tx + 16 * b, N - 1) * bcs;
            #pragma unroll 4
            for (int k = 0; k < Kd; ++k){
                float av[TM], bv[TN];
                #pragma unroll
                for (int a = 0; a < TM; ++a) av[a] = ToF(A[ai[a] + k * acs]);
                #pragma unroll
                for (int b = 0; b < TN; ++b) bv[b] = ToF(B[k * brs + bj[b]]);
                #pragma unroll
                for (int a = 0; a < TM; ++a)
                    #pragma unroll
                    for (int b = 0; b < TN; ++b) acc[a][b] = fmaf(av[a], bv[b], acc[a][b]);
            }
            #pragma unroll
            for (int a = 0; a < TM; ++a){
                const int i = i0 + ty + 16 * a;
                #pragma unroll
                for (int b = 0; b < TN; ++b){
                    const int j = j0 + tx + 16 * b;
                    if (i < M && j < N) epi(i, j, acc[a][b]);
                }
            }
        }
    }
}

// 序列 / head 维按 16 的倍数切块：L <= 80 时 TM = 5 一次覆盖所有行
constexpr int ATM = 5, ATN = 4;

template <typename T>
__global__ void __launch_bounds__(256) AttnFwd(const T* __restrict__ qkv, T* __restrict__ out, float* __restrict__ probs,
                                               int L, int H, int hd, float scale){
    extern __shared__ float sh[];
    const int ld = hd + RowPad<T>(), lds = L + 1;
    float* S = sh;                              // [L][lds] fp32
    T* Q = (T*)(S + L * lds);
    T* K = Q + L * ld;
    T* V = K + L * ld;
    const int b = blockIdx.x / H, h = blockIdx.x % H;
    LoadHead(qkv, Q, b, 0, h, L, H, hd, ld);
    LoadHead(qkv, K, b, 1, h, L, H, hd, ld);
    LoadHead(qkv, V, b, 2, h, L, H, hd, ld);
    __syncthreads();
    // S = scale * Q K^T
    BlockGemm<ATM, ATM>(Q, ld, 1, K, 1, ld, L, L, hd, [&](int i, int j, float v){ S[i * lds + j] = v * scale; });
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    float* P = probs + (size_t)blockIdx.x * L * L;
    for (int i = warp; i < L; i += warps){
        float mx = -INFINITY;
        for (int j = lane; j < L; j += 32) mx = fmaxf(mx, S[i * lds + j]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float sum = 0.f;
        for (int j = lane; j < L; j += 32){ float e = __expf(S[i * lds + j] - mx); S[i * lds + j] = e; sum += e; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        const float inv = 1.f / sum;
        for (int j = lane; j < L; j += 32){ float p = S[i * lds + j] * inv; S[i * lds + j] = p; P[i * L + j] = p; }
    }
    __syncthreads();
    // O = P V
    BlockGemm<ATM, ATN>(S, lds, 1, V, ld, 1, L, hd, L, [&](int i, int d, float v){
        out[(((size_t)b * L + i) * H + h) * hd + d] = FromF<T>(v);
    });
}

template <typename T>
__global__ void __launch_bounds__(256) AttnBwd(const T* __restrict__ qkv, const float* __restrict__ probs,
                                               const T* __restrict__ dout, T* __restrict__ dqkv, int L, int H, int hd,
                                               float scale){
    extern __shared__ float sh[];
    const int ld = hd + RowPad<T>(), lds = L + 1;
    float* r = sh;               // [L]
    T* Q = (T*)(r + L);
    T* K = Q + L * ld;
    T* V = K + L * ld;
    T* dO = V + L * ld;
    T* P = dO + L * ld;          // 先存 P，之后原地改成 dS（bf16 时与 flash attention 一样用 bf16 的 P / dS 做矩阵乘）
    const int b = blockIdx.x / H, h = blockIdx.x % H;
    LoadHead(qkv, Q, b, 0, h, L, H, hd, ld);
    LoadHead(qkv, K, b, 1, h, L, H, hd, ld);
    LoadHead(qkv, V, b, 2, h, L, H, hd, ld);
    LoadRows(dout + (size_t)b * L * H * hd + (size_t)h * hd, (size_t)H * hd, dO, L, hd, ld);
    const float* Pg = probs + (size_t)blockIdx.x * L * L;
    {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int i = warp; i < L; i += blockDim.x >> 5)
            for (int j = lane; j < L; j += 32) P[i * lds + j] = FromF<T>(Pg[i * L + j]);
    }
    for (int i = threadIdx.x; i < L; i += blockDim.x) r[i] = 0.f;
    __syncthreads();
    auto dst = [&](int which, int l, int d) -> T& {
        return dqkv[(((size_t)b * L + l) * 3 + which) * H * hd + (size_t)h * hd + d];
    };
    // dV = P^T dO
    BlockGemm<ATM, ATN>(P, 1, lds, dO, ld, 1, L, hd, L, [&](int j, int d, float v){ dst(2, j, d) = FromF<T>(v); });
    // r_i = dot(dO_i, O_i)，O = P V 重算（每个元素乘上 dO 后 atomicAdd 到共享内存）
    BlockGemm<ATM, ATN>(P, lds, 1, V, ld, 1, L, hd, L, [&](int i, int d, float v){ atomicAdd(&r[i], v * ToF(dO[i * ld + d])); });
    __syncthreads();
    // dS_ij = P_ij (dP_ij - r_i)，dP = dO V^T；每个元素只由一个线程读写
    BlockGemm<ATM, ATM>(dO, ld, 1, V, 1, ld, L, L, hd, [&](int i, int j, float v){
        P[i * lds + j] = FromF<T>(ToF(P[i * lds + j]) * (v - r[i]));
    });
    __syncthreads();
    // dQ = scale dS K，dK = scale dS^T Q
    BlockGemm<ATM, ATN>(P, lds, 1, K, ld, 1, L, hd, L, [&](int i, int d, float v){ dst(0, i, d) = FromF<T>(v * scale); });
    BlockGemm<ATM, ATN>(P, 1, lds, Q, ld, 1, L, hd, L, [&](int j, int d, float v){ dst(1, j, d) = FromF<T>(v * scale); });
}

template <typename T>
size_t FwdSmem(int L, int hd){ return (size_t)L * (L + 1) * sizeof(float) + (size_t)3 * L * (hd + RowPad<T>()) * sizeof(T); }
template <typename T>
size_t BwdSmem(int L, int hd){
    return (size_t)L * sizeof(float) + ((size_t)4 * L * (hd + RowPad<T>()) + (size_t)L * (L + 1)) * sizeof(T);
}

}  // namespace

template <typename T>
void AttentionForward(const TinyTensor<T>& qkv, int heads, TinyTensor<T>& out, TinyTensor<float>& probs){
    NNCheck(qkv.shape.size() == 5 && qkv.shape[2] == 3 && qkv.shape[3] == heads, "attention: qkv [B, L, 3, H, hd]");
    const int B = qkv.shape[0], L = qkv.shape[1], hd = qkv.shape[4];
    NNCheck(L <= 128, "attention: sequence too long for the shared-memory kernel");
    NNCheck(hd % 2 == 0, "attention: head dim must be even");
    const size_t smem = FwdSmem<T>(L, hd);
    NNCheck(smem <= 98 * 1024 && BwdSmem<T>(L, hd) <= 98 * 1024, "attention: head too large for shared memory");
    static bool configured = false;
    if (!configured){
        cudaFuncSetAttribute(AttnFwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024);
        cudaFuncSetAttribute(AttnBwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98 * 1024);
        configured = true;
    }
    out.resize_discard({B, L, heads, hd});
    if constexpr (std::is_same<T, bf16>::value){
        if (AttentionTcSupported(L, hd)){
            probs.resize_discard({B, heads, L});
            AttentionTcForward(qkv.p_data, out.p_data, probs.p_data, B, L, heads);
            return;
        }
    }
    probs.resize_discard({B, heads, L, L});
    AttnFwd<T><<<B * heads, 256, smem>>>(qkv.p_data, out.p_data, probs.p_data, L, heads, hd, 1.f / sqrtf((float)hd));
}

template <typename T>
void AttentionBackward(const TinyTensor<T>& qkv, const TinyTensor<float>& probs, const TinyTensor<T>& dout, int heads,
                       TinyTensor<T>& dqkv){
    const int B = qkv.shape[0], L = qkv.shape[1], hd = qkv.shape[4];
    dqkv.resize_discard(qkv.shape);
    if constexpr (std::is_same<T, bf16>::value){
        if (AttentionTcSupported(L, hd)){
            AttentionTcBackward(qkv.p_data, probs.p_data, dout.p_data, dqkv.p_data, B, L, heads);
            return;
        }
    }
    AttnBwd<T><<<B * heads, 256, BwdSmem<T>(L, hd)>>>(qkv.p_data, probs.p_data, dout.p_data, dqkv.p_data, L, heads, hd,
                                                  1.f / sqrtf((float)hd));
}

template void AttentionForward<float>(const TinyTensor<float>&, int, TinyTensor<float>&, TinyTensor<float>&);
template void AttentionForward<bf16>(const TinyTensor<bf16>&, int, TinyTensor<bf16>&, TinyTensor<float>&);
template void AttentionBackward<float>(const TinyTensor<float>&, const TinyTensor<float>&, const TinyTensor<float>&, int,
                                       TinyTensor<float>&);
template void AttentionBackward<bf16>(const TinyTensor<bf16>&, const TinyTensor<float>&, const TinyTensor<bf16>&, int,
                                      TinyTensor<bf16>&);
