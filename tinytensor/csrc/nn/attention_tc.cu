// bf16 多头注意力的 tensor core 版本（mma.sync.m16n8k16，fp32 累加），适用于短序列：L <= 80、head 维 64（ViT-Tiny）
// 一个 block（5 个 warp）负责一个 (batch, head)，序列补到 80 行（不足的行在共享内存里填 0）
// 前向：warp w 负责第 16w ~ 16w+15 个 query
//   S = Q K^T 留在寄存器里（16 x 80，每个线程 40 个），按行 softmax（同一行分布在 4 个 lane 上，用 shuffle 归约），
//   P 的累加器布局正好就是下一个 mma 的 A 操作数布局，直接转成 bf16 算 O = P V；只保存每行的 logsumexp
// 反向：
//   阶段 1（warp 负责 16 个 query）：重算 P = exp(S - lse)，dP = dO V^T，r_i = sum_j P_ij dP_ij，
//     dS = P (dP - r)；dQ = scale * dS K（dS 仍在寄存器里）；P 与 dS 以 bf16 写进共享内存
//   阶段 2（warp 负责 16 个 key）：dV = P^T dO，dK = scale * dS^T Q（ldmatrix.trans 读转置）
// 与 flash attention 一样，P / dS 参与矩阵乘时是 bf16
#include <cmath>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

constexpr int TL = 80, HD = 64, NW = TL / 16, THREADS = NW * 32;
constexpr int QS = HD + 8;      // Q / K / V / dO 的行距（144 字节：ldmatrix 的 8 行落在不同 bank）
constexpr int PS = TL + 8;      // P / dS 的行距（176 字节）

__device__ __forceinline__ unsigned Smem(const void* p){ return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void CpAsync16(void* dst, const void* src, bool valid){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(Smem(dst)), "l"(src), "r"(valid ? 16 : 0));
}
template <bool TRANS>
__device__ __forceinline__ void Ldm4(unsigned* r, const void* p){
    if (TRANS)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(Smem(p)));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(Smem(p)));
}
__device__ __forceinline__ void Mma(float* c, const unsigned* a, unsigned b0, unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned Pack(float lo, float hi){
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *(unsigned*)&h;
}

// ldmatrix 的地址（lane 对应的行 / 列），见 conv_tc.cu：
// A 按行存 [m][k]：行 m0 + (lane & 15)，列 k0 + (lane >> 4) * 8
__device__ __forceinline__ const bf16* AddrA(const bf16* base, int ld, int m0, int k0, int lane){
    return base + (m0 + (lane & 15)) * ld + k0 + ((lane >> 4) << 3);
}
// A 按列存（X[k][m]，用 .trans）：行 k0 + (lane & 7) + (lane >> 4) * 8，列 m0 + ((lane >> 3) & 1) * 8
__device__ __forceinline__ const bf16* AddrAT(const bf16* base, int ld, int m0, int k0, int lane){
    return base + (k0 + (lane & 7) + ((lane >> 4) << 3)) * ld + m0 + (((lane >> 3) & 1) << 3);
}
// B 按 [n][k] 存（两个 n8 tile）：行 n0 + (lane & 7) + (lane >> 4) * 8，列 k0 + ((lane >> 3) & 1) * 8
__device__ __forceinline__ const bf16* AddrB(const bf16* base, int ld, int n0, int k0, int lane){
    return base + (n0 + (lane & 7) + ((lane >> 4) << 3)) * ld + k0 + (((lane >> 3) & 1) << 3);
}
// B 按 [k][n] 存（用 .trans）：行 k0 + (lane & 7) + ((lane >> 3) & 1) * 8，列 n0 + (lane >> 4) * 8
__device__ __forceinline__ const bf16* AddrBT(const bf16* base, int ld, int n0, int k0, int lane){
    return base + (k0 + (lane & 7) + (((lane >> 3) & 1) << 3)) * ld + n0 + ((lane >> 4) << 3);
}

// 把一个 head 的 [L][64] 读进共享内存 [80][QS]，多出的行填 0
__device__ __forceinline__ void LoadHeadAsync(bf16* dst, const bf16* src, size_t row_stride, int L){
    for (int i = threadIdx.x; i < TL * (HD / 8); i += THREADS){
        int l = i >> 3, c = (i & 7) * 8;
        bool ok = l < L;
        CpAsync16(dst + l * QS + c, ok ? src + l * row_stride + c : src, ok);
    }
}
__device__ __forceinline__ void WaitAll(){
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}

// acc[10][4] = Q[m0..m0+15] K^T（16 x 80）
__device__ __forceinline__ void ScoresQK(float (*acc)[4], const bf16* Q, const bf16* K, int m0, int lane){
    #pragma unroll
    for (int j = 0; j < TL / 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
    #pragma unroll
    for (int kk = 0; kk < HD; kk += 16){
        unsigned a[4];
        Ldm4<false>(a, AddrA(Q, QS, m0, kk, lane));
        #pragma unroll
        for (int np = 0; np < TL / 16; ++np){
            unsigned b[4];
            Ldm4<false>(b, AddrB(K, QS, np * 16, kk, lane));
            Mma(acc[2 * np], a, b[0], b[1]);
            Mma(acc[2 * np + 1], a, b[2], b[3]);
        }
    }
}

// out[16 x 64] = P(寄存器里的 16 x 80，fp32 累加器布局) * B，B 按 [k][n] 存在共享内存
__device__ __forceinline__ void PTimes(float (*out)[4], const float (*p)[4], const bf16* B, int lane){
    #pragma unroll
    for (int j = 0; j < HD / 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) out[j][e] = 0.f;
    #pragma unroll
    for (int kk = 0; kk < TL / 16; ++kk){
        unsigned a[4] = {Pack(p[2 * kk][0], p[2 * kk][1]), Pack(p[2 * kk][2], p[2 * kk][3]),
                         Pack(p[2 * kk + 1][0], p[2 * kk + 1][1]), Pack(p[2 * kk + 1][2], p[2 * kk + 1][3])};
        #pragma unroll
        for (int np = 0; np < HD / 16; ++np){
            unsigned b[4];
            Ldm4<true>(b, AddrBT(B, QS, np * 16, kk * 16, lane));
            Mma(out[2 * np], a, b[0], b[1]);
            Mma(out[2 * np + 1], a, b[2], b[3]);
        }
    }
}

// 把 16 x 64 的 fp32 累加器写到全局：行 i 写到 dst + i * row_stride
__device__ __forceinline__ void StoreRows(bf16* dst, size_t row_stride, const float (*acc)[4], int m0, int L, int lane,
                                          float scale){
    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int half = 0; half < 2; ++half){
        const int i = m0 + g + half * 8;
        if (i >= L) continue;
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j)
            *(__nv_bfloat162*)(dst + i * row_stride + j * 8 + 2 * t) =
                __floats2bfloat162_rn(acc[j][half * 2] * scale, acc[j][half * 2 + 1] * scale);
    }
}

__global__ void __launch_bounds__(THREADS) AttnTcFwd(const bf16* __restrict__ qkv, bf16* __restrict__ out,
                                                     float* __restrict__ lse, int L, int H, float scale){
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* Q = (bf16*)smem_raw;
    bf16* K = Q + TL * QS;
    bf16* V = K + TL * QS;
    const int b = blockIdx.x / H, h = blockIdx.x % H, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t rs = (size_t)3 * H * HD;
    const bf16* base = qkv + (size_t)b * L * rs + (size_t)h * HD;
    LoadHeadAsync(Q, base, rs, L);
    LoadHeadAsync(K, base + H * HD, rs, L);
    LoadHeadAsync(V, base + 2 * H * HD, rs, L);
    WaitAll();

    const int m0 = warp * 16, g = lane >> 2, t = lane & 3;
    float s[TL / 8][4];
    ScoresQK(s, Q, K, m0, lane);
    // 按行 softmax：本线程持有第 g 行（e = 0, 1）与第 g + 8 行（e = 2, 3）
    float mx[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int j = 0; j < TL / 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e){
            const int col = j * 8 + 2 * t + (e & 1);
            s[j][e] = col < L ? s[j][e] * scale : -INFINITY;
            mx[e >> 1] = fmaxf(mx[e >> 1], s[j][e]);
        }
    float sum[2] = {0.f, 0.f};
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh){
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
    }
    #pragma unroll
    for (int j = 0; j < TL / 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e){
            s[j][e] = __expf(s[j][e] - mx[e >> 1]);
            sum[e >> 1] += s[j][e];
        }
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh){
        sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 1);
        sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 2);
        const int i = m0 + g + hh * 8;
        if (t == 0 && i < L) lse[(size_t)blockIdx.x * L + i] = mx[hh] + __logf(sum[hh]);
    }
    const float inv[2] = {1.f / sum[0], 1.f / sum[1]};
    #pragma unroll
    for (int j = 0; j < TL / 8; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) s[j][e] *= inv[e >> 1];

    float o[HD / 8][4];
    PTimes(o, s, V, lane);
    StoreRows(out + ((size_t)b * L * H + h) * HD, (size_t)H * HD, o, m0, L, lane, 1.f);
}

__global__ void __launch_bounds__(THREADS) AttnTcBwd(const bf16* __restrict__ qkv, const float* __restrict__ lse,
                                                     const bf16* __restrict__ dout, bf16* __restrict__ dqkv,
                                                     int L, int H, float scale){
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* Q = (bf16*)smem_raw;
    bf16* K = Q + TL * QS;
    bf16* V = K + TL * QS;
    bf16* dO = V + TL * QS;
    bf16* Ps = dO + TL * QS;       // [80][PS]
    bf16* dSs = Ps + TL * PS;      // [80][PS]
    const int b = blockIdx.x / H, h = blockIdx.x % H, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t rs = (size_t)3 * H * HD;
    const bf16* base = qkv + (size_t)b * L * rs + (size_t)h * HD;
    LoadHeadAsync(Q, base, rs, L);
    LoadHeadAsync(K, base + H * HD, rs, L);
    LoadHeadAsync(V, base + 2 * H * HD, rs, L);
    LoadHeadAsync(dO, dout + (size_t)b * L * H * HD + (size_t)h * HD, (size_t)H * HD, L);
    WaitAll();

    const int m0 = warp * 16, g = lane >> 2, t = lane & 3;
    bf16* dbase = dqkv + (size_t)b * L * rs + (size_t)h * HD;
    {
        // ---- 阶段 1：本 warp 的 16 个 query ----
        float p[TL / 8][4];
        ScoresQK(p, Q, K, m0, lane);
        float l2[2];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh){
            const int i = m0 + g + hh * 8;
            l2[hh] = i < L ? lse[(size_t)blockIdx.x * L + i] : INFINITY;   // 补出来的行 P = 0
        }
        #pragma unroll
        for (int j = 0; j < TL / 8; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e){
                const int col = j * 8 + 2 * t + (e & 1);
                p[j][e] = col < L ? __expf(p[j][e] * scale - l2[e >> 1]) : 0.f;
            }
        // dP = dO V^T（V 按 [key][d] 存，正好是 B 的 [n][k]）
        float dp[TL / 8][4];
        ScoresQK(dp, dO, V, m0, lane);
        float r[2] = {0.f, 0.f};
        #pragma unroll
        for (int j = 0; j < TL / 8; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) r[e >> 1] += p[j][e] * dp[j][e];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh){
            r[hh] += __shfl_xor_sync(0xffffffff, r[hh], 1);
            r[hh] += __shfl_xor_sync(0xffffffff, r[hh], 2);
        }
        // dS = P (dP - r)；P、dS 写进共享内存供阶段 2 使用
        #pragma unroll
        for (int j = 0; j < TL / 8; ++j){
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh){
                const int row = m0 + g + hh * 8, col = j * 8 + 2 * t;
                dp[j][hh * 2] = p[j][hh * 2] * (dp[j][hh * 2] - r[hh]);
                dp[j][hh * 2 + 1] = p[j][hh * 2 + 1] * (dp[j][hh * 2 + 1] - r[hh]);
                *(__nv_bfloat162*)(Ps + row * PS + col) = __floats2bfloat162_rn(p[j][hh * 2], p[j][hh * 2 + 1]);
                *(__nv_bfloat162*)(dSs + row * PS + col) = __floats2bfloat162_rn(dp[j][hh * 2], dp[j][hh * 2 + 1]);
            }
        }
        // dQ = scale * dS K（K 按 [key][d] 存，是 B 的 [k][n]）
        float dq[HD / 8][4];
        PTimes(dq, dp, K, lane);
        StoreRows(dbase, rs, dq, m0, L, lane, scale);
    }
    __syncthreads();
    {
        // ---- 阶段 2：本 warp 的 16 个 key：dV = P^T dO，dK = scale * dS^T Q ----
        float dv[HD / 8][4], dk[HD / 8][4];
        #pragma unroll
        for (int j = 0; j < HD / 8; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e){ dv[j][e] = 0.f; dk[j][e] = 0.f; }
        #pragma unroll
        for (int kk = 0; kk < TL; kk += 16){
            unsigned ap[4], as[4];
            Ldm4<true>(ap, AddrAT(Ps, PS, m0, kk, lane));
            Ldm4<true>(as, AddrAT(dSs, PS, m0, kk, lane));
            #pragma unroll
            for (int np = 0; np < HD / 16; ++np){
                unsigned bo[4], bq[4];
                Ldm4<true>(bo, AddrBT(dO, QS, np * 16, kk, lane));
                Ldm4<true>(bq, AddrBT(Q, QS, np * 16, kk, lane));
                Mma(dv[2 * np], ap, bo[0], bo[1]);
                Mma(dv[2 * np + 1], ap, bo[2], bo[3]);
                Mma(dk[2 * np], as, bq[0], bq[1]);
                Mma(dk[2 * np + 1], as, bq[2], bq[3]);
            }
        }
        StoreRows(dbase + H * HD, rs, dk, m0, L, lane, scale);
        StoreRows(dbase + 2 * H * HD, rs, dv, m0, L, lane, 1.f);
    }
}

constexpr size_t FWD_SMEM = (size_t)3 * TL * QS * sizeof(bf16);
constexpr size_t BWD_SMEM = ((size_t)4 * TL * QS + (size_t)2 * TL * PS) * sizeof(bf16);

}  // namespace

bool AttentionTcSupported(int L, int hd){ return L <= TL && hd == HD; }

static void Configure(){
    static bool configured = false;
    if (!configured){
        cudaFuncSetAttribute(AttnTcFwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FWD_SMEM);
        cudaFuncSetAttribute(AttnTcBwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)BWD_SMEM);
        configured = true;
    }
}

void AttentionTcForward(const bf16* qkv, bf16* out, float* lse, int B, int L, int H){
    Configure();
    AttnTcFwd<<<B * H, THREADS, FWD_SMEM>>>(qkv, out, lse, L, H, 1.f / sqrtf((float)HD));
}

void AttentionTcBackward(const bf16* qkv, const float* lse, const bf16* dout, bf16* dqkv, int B, int L, int H){
    Configure();
    AttnTcBwd<<<B * H, THREADS, BWD_SMEM>>>(qkv, lse, dout, dqkv, L, H, 1.f / sqrtf((float)HD));
}
