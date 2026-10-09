// bf16 卷积的 tensor core 版本：隐式 GEMM + mma.sync.m16n8k16（fp32 累加）
// - 全局 -> 共享内存用 cp.async（16 字节 = 8 个 bf16，越界的块由硬件填 0），3 或 4 级流水（按每个 SM 的共享内存选，见 Stages）
// - 共享内存 -> 寄存器用 ldmatrix；按 K 连续存放的 tile 用普通 ldmatrix，按 M/N 连续存放的用 ldmatrix.trans
// - 共享内存按 16 字节块做 XOR swizzle，ldmatrix 与 cp.async 都没有 bank conflict
// 三种 GEMM 的定义与 conv.cu 相同：
//   FWD  : A = im2col(x) [M=N*P*Q][K=R*S*C/g]（K 连续）, B = w [N=K/g][K]（K 连续）
//   DGRAD: A = 反向 im2col(dy) [M=N*H*W][K=R*S*K/g]（K 连续）, B = w 按 (r,s,k) 取 [K][N=C/g]（N 连续）
//          步长 > 1 时按 (ih % sh, iw % sw) 把输入像素分成 sh*sw 个相位，每个 block 只含一个相位的像素，
//          只循环对这个相位有效的抽头（步长 2 的 3x3 卷积省掉 3/4 的无效计算）
//   WGRAD: A = dy^T [M=K/g][K=像素]（M 连续）, B = im2col(x) [K=像素][N=R*S*C/g]（N 连续），split-K + atomicAdd
// 只处理通道数对齐的情况（见 ConvTensorCoreSupported），其余回退到 conv.cu 的 SIMT 版本
#include <cstdlib>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

enum Mode { FWD = 0, DGRAD = 1, WGRAD = 2 };
constexpr int BK = 32;

struct TcArgs{
    ConvGeom g;
    int cg, kg, gm, gn, gk;
    int k_per_split, splits;
    int bpp;             // DGRAD：每个相位的 block 数（gm 是每个相位的像素数）
};

__device__ __forceinline__ unsigned Smem(const void* p){ return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void CpAsync16(void* dst, const void* src, bool valid){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(Smem(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void CpCommit(){ asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void CpWait(){ asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

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

// tile 内的元素偏移（单位：bf16）
// RK：[rows][BK=32]，每行 4 个 16B 块，块号与 (row>>1)&3 异或
__device__ __forceinline__ int OffRK(int row, int k){ return row * BK + ((((k >> 3) ^ (row >> 1)) & 3) << 3) + (k & 7); }
// KR：[BK][ROWS]，每行 ROWS/8 个块，块号与 k&7 异或（ROWS >= 64）
template <int ROWS>
__device__ __forceinline__ int OffKR(int k, int col){ return k * ROWS + ((((col >> 3) ^ k) & 7) | ((col >> 3) & ~7)) * 8 + (col & 7); }

// A 是否按 K 连续（RK 布局）：FWD / DGRAD 是，WGRAD 不是（KR 布局）
template <int MODE> struct Layout{ static constexpr bool A_RK = MODE != WGRAD; static constexpr bool B_RK = MODE == FWD; };

// block tile BM x BN，WGM x WGN 个 warp，每个 warp 负责 (BM/WGM) x (BN/WGN)。
// 配置见 Dispatch：RTX 40 系列用 8 个 warp 的 128x128 / 128x64；A100 这类共享内存大的卡用 4 个 warp、每个 warp 64x64 的大 tile
// （和 cuDNN 在 A100 上选的 256x64 / 128x128 一致：每个 k16 步 MMA 与 ldmatrix 之比更高）
template <int MODE, int BM, int BN, int STAGES, int WGM, int WGN>
__global__ void __launch_bounds__(WGM * WGN * 32, 2) ConvTc(const bf16* __restrict__ a, const bf16* __restrict__ b, void* __restrict__ out,
                                                  const TcArgs p){
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* As = (bf16*)smem_raw;                       // [STAGES][BM * BK]
    bf16* Bs = As + STAGES * BM * BK;                 // [STAGES][BN * BK]
    constexpr bool A_RK = Layout<MODE>::A_RK, B_RK = Layout<MODE>::B_RK;
    constexpr int THREADS = WGM * WGN * 32;
    constexpr int WM = BM / WGM, MI = WM / 16;        // 每个 warp WM x WN，MI 个 m16 tile
    constexpr int WN = BN / WGN, NT = WN / 8;         // NT 个 n8 tile
    constexpr int A_CHUNKS = BM * BK / 8 / THREADS;   // 每个线程每级搬运的 16 字节块数
    constexpr int B_CHUNKS = BN * BK / 8 / THREADS;
    static_assert(A_CHUNKS * THREADS * 8 == BM * BK && B_CHUNKS * THREADS * 8 == BN * BK && NT % 2 == 0, "tile 配置");

    const ConvGeom& g = p.g;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WGM, wn = warp / WGM;
    int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int group = blockIdx.z / p.splits, split = blockIdx.z % p.splits;
    const int k_begin = split * p.k_per_split;
    const int k_end = min(p.gk, k_begin + p.k_per_split);
    int ktiles = (k_end - k_begin + BK - 1) / BK;
    // DGRAD：本 block 的相位 (ph_h, ph_w)，有效抽头 r = r0 + i*sh, s = s0 + j*sw
    int ph_h = 0, ph_w = 0, r0 = 0, s0 = 0, ns = g.s;
    const int Hp = g.h / g.stride_h, Wp = g.w / g.stride_w;
    if (MODE == DGRAD){
        const int phase = blockIdx.x / p.bpp;
        m0 = (blockIdx.x % p.bpp) * BM;
        ph_h = phase / g.stride_w; ph_w = phase % g.stride_w;
        r0 = (ph_h + g.pad_h) % g.stride_h; s0 = (ph_w + g.pad_w) % g.stride_w;
        const int nr = r0 < g.r ? (g.r - r0 + g.stride_h - 1) / g.stride_h : 0;
        ns = s0 < g.s ? (g.s - s0 + g.stride_w - 1) / g.stride_w : 0;
        ktiles = nr * ns * (p.kg / BK);
    }

    // FWD / DGRAD 的 A 每个线程固定负责 A_CHUNKS 行：预先把行号拆成 (n, h, w)。
    // h、w 压进一个 int 的高低 16 位（省寄存器，大 tile 每个线程有 8 行）；越界的行 h 取 -16384，后面的边界检查必然不过
    int a_n[A_CHUNKS], a_hw[A_CHUNKS];
    if (A_RK){
        #pragma unroll
        for (int i = 0; i < A_CHUNKS; ++i){
            int m = m0 + ((tid + i * THREADS) >> 2);
            const bool ok = m < p.gm;
            int mm = ok ? m : 0, h, w;
            if (MODE == FWD){
                int ow = mm % g.q; int t = mm / g.q; int oh = t % g.p;
                h = oh * g.stride_h - g.pad_h; w = ow * g.stride_w - g.pad_w;
                a_n[i] = (t / g.p * g.h + h) * g.w + w;   // FWD：存左上角像素的下标（可能越界，只在检查通过后使用）
            }
            else{
                // 第 (ti, tj) 个有效抽头对应 dy 的 (oh, ow) = (ih + ch - ti, iw + cw - tj)，ch = (ph_h + pad_h - r0) / sh 整除
                int iw = mm % Wp; int t = mm / Wp; int ih = t % Hp;
                h = ih + (ph_h + g.pad_h - r0) / g.stride_h; w = iw + (ph_w + g.pad_w - s0) / g.stride_w;
                a_n[i] = (t / Hp * g.p + h) * g.q + w;
            }
            a_hw[i] = (ok ? h : -16384) * 65536 + (w & 0xffff);
        }
    }

    // WGRAD 的 B：每个线程负责的列 n 固定，(r, s, c) 只需拆一次
    int wb_c = 0, wb_r = 0, wb_s = 0;
    bool wb_ok = false;
    if (MODE == WGRAD){
        int n = n0 + (tid % (BN / 8)) * 8;
        wb_ok = n < p.gn;
        int nn = wb_ok ? n : 0;
        wb_c = nn % p.cg; int t = nn / p.cg; wb_s = t % g.s; wb_r = t / g.s;
        wb_r -= g.pad_h; wb_s -= g.pad_w;
    }

    // 第 kt 个 K 方向的 tile：FWD / WGRAD 是 [k0, k0 + BK)；DGRAD 是第 tap 个有效抽头的第 cbase 个通道起的 BK 个通道
    auto load_tile = [&](int stage, int kt){
        const int k0 = k_begin + kt * BK;
        int tap_r = 0, tap_s = 0, tap_c = 0, tap_i = 0, tap_j = 0;
        if (MODE == DGRAD){
            const int tpt = p.kg / BK, ti = kt / tpt;
            tap_c = (kt % tpt) * BK;
            tap_i = ti / ns; tap_j = ti % ns;
            tap_r = r0 + tap_i * g.stride_h; tap_s = s0 + tap_j * g.stride_w;
        }
        bf16* as = As + stage * BM * BK;
        bf16* bs = Bs + stage * BN * BK;
        // ---- A ----
        if (MODE == FWD || MODE == DGRAD){
            // 整个 BK 段落在同一个抽头 (r, s) 内（通道数是 32 的倍数）
            const int cbase = MODE == FWD ? k0 % p.cg : tap_c;
            const int rs = k0 / p.cg;
            const int s = MODE == FWD ? rs % g.s : tap_s, r = MODE == FWD ? rs / g.s : tap_r;
            const int rws = MODE == FWD ? r * g.w + s : -(tap_i * g.q + tap_j);
            #pragma unroll
            for (int i = 0; i < A_CHUNKS; ++i){
                int id = tid + i * THREADS, row = id >> 2, kc = id & 3;
                const bf16* src = a;
                const int ah = a_hw[i] >> 16, aw = (short)(a_hw[i] & 0xffff);
                bool ok;
                if (MODE == FWD){
                    int ih = ah + r, iw = aw + s;
                    ok = ih >= 0 && ih < g.h && iw >= 0 && iw < g.w;
                    if (ok) src = a + (size_t)(a_n[i] + rws) * g.c + group * p.cg + cbase + kc * 8;
                }
                else{
                    int oh = ah - tap_i, ow = aw - tap_j;
                    ok = oh >= 0 && oh < g.p && ow >= 0 && ow < g.q;
                    if (ok) src = a + (size_t)(a_n[i] + rws) * g.k + group * p.kg + cbase + kc * 8;
                }
                CpAsync16(as + OffRK(row, kc * 8), src, ok);
            }
        }
        else{
            // WGRAD：A(m = 输出通道, kk = 像素) = dy[像素][通道]，沿 m 连续
            #pragma unroll
            for (int i = 0; i < A_CHUNKS; ++i){
                constexpr int MCH = BM / 8;
                int id = tid + i * THREADS, krow = id / MCH, mc = id % MCH;
                int pix = k0 + krow, m = m0 + mc * 8;
                bool ok = pix < k_end && m < p.gm;
                const bf16* src = ok ? a + (size_t)pix * g.k + group * p.kg + m : a;
                CpAsync16(as + OffKR<BM>(krow, mc * 8), src, ok);
            }
        }
        // ---- B ----
        #pragma unroll
        for (int i = 0; i < B_CHUNKS; ++i){
            int id = tid + i * THREADS;
            if (MODE == FWD){
                int row = id >> 2, kc = id & 3, n = n0 + row;
                bool ok = n < p.gn;
                const bf16* src = ok ? b + (size_t)(group * p.kg + n) * p.gk + k0 + kc * 8 : b;
                CpAsync16(bs + OffRK(row, kc * 8), src, ok);
            }
            else if (MODE == DGRAD){
                // B(kk = (r, s, k), n = c) = w[k][r][s][c]
                constexpr int CPR = BN / 8;
                int krow = id / CPR, nc = id % CPR, n = n0 + nc * 8;
                int k = tap_c + krow;
                bool ok = n < p.gn;
                const bf16* src = ok ? b + (((size_t)(group * p.kg + k) * g.r + tap_r) * g.s + tap_s) * p.cg + n : b;
                CpAsync16(bs + OffKR<BN>(krow, nc * 8), src, ok);
            }
            else{
                // B(kk = 像素, n = (r, s, c)) = x[n_img][oh*sh - ph + r][ow*sw - pw + s][c]
                constexpr int CPR = BN / 8;
                int krow = id / CPR, nc = id % CPR;
                int pix = k0 + krow;
                bool ok = pix < k_end && wb_ok;
                const bf16* src = b;
                if (ok){
                    int ow = pix % g.q; int t = pix / g.q; int oh = t % g.p; int ni = t / g.p;
                    int ih = oh * g.stride_h + wb_r, iw = ow * g.stride_w + wb_s;
                    ok = ih >= 0 && ih < g.h && iw >= 0 && iw < g.w;
                    if (ok) src = b + (((size_t)ni * g.h + ih) * g.w + iw) * g.c + group * p.cg + wb_c;
                }
                CpAsync16(bs + OffKR<BN>(krow, nc * 8), src, ok);
            }
        }
    };

    float acc[MI][NT][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // 寄存器里的 fragment 双缓冲：做第 ks 个 k16 的 MMA 前先发出第 ks + 1 个 k16 的 ldmatrix；
    // tile 的最后一个 k16 之前过 barrier，接着就读下一个 tile 的第一组 fragment，barrier 后马上有 MMA 可发射。
    // 读最后一组 fragment 之前，本 tile 的共享内存已经全部读完，所以 barrier 后可以直接覆盖它（预取 STAGES 个 tile）
    // 寄存器放不下两份 fragment 时（8 个 warp 的 128x128：每线程上限 128 个寄存器）退回单缓冲：每个 k16 先读再算
    constexpr int KS = BK / 16;
    static_assert(KS % 2 == 0, "BK 必须是 32 的倍数");
    constexpr int MAX_REGS = 65536 / 2 / THREADS > 255 ? 255 : 65536 / 2 / THREADS;
    constexpr bool FRAG_DB = MI * NT * 4 + 2 * (MI * 4 + NT * 2) <= MAX_REGS - 40;
    unsigned af[FRAG_DB ? 2 : 1][MI][4], bfr[FRAG_DB ? 2 : 1][NT][2];
    auto load_frag = [&](int buf, int stage, int kk){
        const bf16* as = As + stage * BM * BK;
        const bf16* bs = Bs + stage * BN * BK;
        #pragma unroll
        for (int i = 0; i < MI; ++i){
            int mbase = wm * WM + i * 16;
            if (A_RK) Ldm4<false>(af[buf][i], as + OffRK(mbase + (lane & 15), kk + ((lane >> 4) << 3)));
            else Ldm4<true>(af[buf][i], as + OffKR<BM>(kk + (lane & 7) + ((lane >> 4) << 3), mbase + (((lane >> 3) & 1) << 3)));
        }
        #pragma unroll
        for (int j = 0; j < NT; j += 2){
            int nbase = wn * WN + j * 8;
            unsigned r[4];
            if (B_RK) Ldm4<false>(r, bs + OffRK(nbase + (lane & 7) + ((lane >> 4) << 3), kk + (((lane >> 3) & 1) << 3)));
            else Ldm4<true>(r, bs + OffKR<BN>(kk + (lane & 7) + (((lane >> 3) & 1) << 3), nbase + ((lane >> 4) << 3)));
            bfr[buf][j][0] = r[0]; bfr[buf][j][1] = r[1]; bfr[buf][j + 1][0] = r[2]; bfr[buf][j + 1][1] = r[3];
        }
    };

    if constexpr (!FRAG_DB){
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s){
            if (s < ktiles) load_tile(s, s);
            CpCommit();
        }
        for (int kt = 0; kt < ktiles; ++kt){
            CpWait<STAGES - 2>();
            __syncthreads();
            int nt = kt + STAGES - 1;
            if (nt < ktiles) load_tile(nt % STAGES, nt);
            CpCommit();
            #pragma unroll
            for (int ks = 0; ks < KS; ++ks){
                load_frag(0, kt % STAGES, ks * 16);
                #pragma unroll
                for (int i = 0; i < MI; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j) Mma(acc[i][j], af[0][i], bfr[0][j][0], bfr[0][j][1]);
            }
        }
        CpWait<0>();
    }
    else{
        #pragma unroll
        for (int s = 0; s < STAGES; ++s){
            if (s < ktiles) load_tile(s, s);
            CpCommit();
        }
        CpWait<STAGES - 1>();
        __syncthreads();
        load_frag(0, 0, 0);

        for (int kt = 0; kt < ktiles; ++kt){
            const int stage = kt % STAGES;
            #pragma unroll
            for (int ks = 0; ks < KS; ++ks){
                if (ks < KS - 1) load_frag((ks + 1) & 1, stage, (ks + 1) * 16);
                else{
                    // 等第 kt + 1 个 tile 到齐；本 tile 已读完，发出第 kt + STAGES 个 tile 覆盖它
                    CpWait<STAGES - 2>();
                    __syncthreads();
                    if (kt + STAGES < ktiles) load_tile(stage, kt + STAGES);
                    CpCommit();
                    load_frag(0, (kt + 1) % STAGES, 0);
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j) Mma(acc[i][j], af[ks & 1][i], bfr[ks & 1][j][0], bfr[ks & 1][j][1]);
            }
        }
        CpWait<0>();
    }

    // ---- 写回 ----
    #pragma unroll
    for (int i = 0; i < MI; ++i){
        #pragma unroll
        for (int half = 0; half < 2; ++half){
            int m = m0 + wm * WM + i * 16 + (lane >> 2) + half * 8;
            if (m >= p.gm) continue;
            size_t row = m;
            if (MODE == DGRAD){
                int iw = m % Wp; int t = m / Wp; int ih = t % Hp; int n = t / Hp;
                row = ((size_t)n * g.h + ih * g.stride_h + ph_h) * g.w + iw * g.stride_w + ph_w;
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j){
                int n = n0 + wn * WN + j * 8 + (lane & 3) * 2;
                float v0 = acc[i][j][half * 2], v1 = acc[i][j][half * 2 + 1];
                if (MODE == WGRAD){
                    float* dw = (float*)out + (size_t)(group * p.kg + m) * p.gn + n;
                    if (n < p.gn) atomicAdd(dw, v0);
                    if (n + 1 < p.gn) atomicAdd(dw + 1, v1);
                }
                else{
                    size_t ld = MODE == FWD ? g.k : g.c;
                    int col = (MODE == FWD ? group * p.kg : group * p.cg) + n;
                    bf16* o = (bf16*)out + row * ld + col;
                    if (n + 1 < p.gn) *(__nv_bfloat162*)o = __floats2bfloat162_rn(v0, v1);
                    else if (n < p.gn) *o = __float2bfloat16(v0);
                }
            }
        }
    }
}

template <int MODE, int BM, int BN, int STAGES, int WGM, int WGN>
void LaunchTc(const bf16* a, const bf16* b, void* out, TcArgs p){
    const int smem = STAGES * (BM + BN) * BK * (int)sizeof(bf16);
    static bool configured = false;
    if (!configured){
        cudaFuncSetAttribute(ConvTc<MODE, BM, BN, STAGES, WGM, WGN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    p.bpp = (p.gm + BM - 1) / BM;
    int gx = p.bpp;
    if (MODE == DGRAD) gx *= p.g.stride_h * p.g.stride_w;
    dim3 grid(gx, (p.gn + BN - 1) / BN, p.g.groups * p.splits);
    ConvTc<MODE, BM, BN, STAGES, WGM, WGN><<<grid, WGM * WGN * 32, smem>>>(a, b, out, p);
}

TcArgs MakeTcArgs(const ConvGeom& g, int mode){
    TcArgs p;
    p.g = g;
    p.cg = g.c / g.groups;
    p.kg = g.k / g.groups;
    if (mode == FWD){ p.gm = g.n * g.p * g.q; p.gn = p.kg; p.gk = g.r * g.s * p.cg; }
    else if (mode == DGRAD){ p.gm = g.n * (g.h / g.stride_h) * (g.w / g.stride_w); p.gn = p.cg; p.gk = g.r * g.s * p.kg; }
    else { p.gm = p.kg; p.gn = g.r * g.s * p.cg; p.gk = g.n * g.p * g.q; }
    p.splits = 1;
    p.k_per_split = p.gk;
    p.bpp = 1;
    return p;
}

int DeviceAttr(cudaDeviceAttr attr){
    int dev = 0, v = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, attr, dev);
    return v;
}

// 流水级数：128x128 的 tile 每级 16KB。每个 SM 的共享内存放得下两个 4 级 block（A100 / H100：164KB 以上）就用 4 级，
// 否则（如 RTX 40 系列每个 SM 100KB）用 3 级，保证每个 SM 能同时跑 2 个 block。环境变量 TT_TC_STAGES 可强制指定（对照实验用）
int Stages(){
    static int st = [](){
        const char* e = getenv("TT_TC_STAGES");
        if (e && (atoi(e) == 3 || atoi(e) == 4)) return atoi(e);
        return DeviceAttr(cudaDevAttrMaxSharedMemoryPerMultiprocessor) >= 2 * 4 * 256 * BK * 2 + 4096 ? 4 : 3;
    }();
    return st;
}

template <int MODE, int BM, int BN, int WGM = 2, int WGN = 4>
void LaunchStages(const bf16* a, const bf16* b, void* out, const TcArgs& p){
    if (Stages() == 4) LaunchTc<MODE, BM, BN, 4, WGM, WGN>(a, b, out, p);
    else LaunchTc<MODE, BM, BN, 3, WGM, WGN>(a, b, out, p);
}

// 大 warp tile 配置（每个 warp 64x64 或 32x64，4 个 warp）：0 = 不用，1 = 只用在 N 只有 64 的层（256x64）和
// WGRAD 输出通道只有 64 的层（64x128），2 = 全部用。A100 上 tune2 实测：256x64 / 64x128 在 VGG 64@64 上快 10~25%，
// 128x128 的 2x2 布局在 128 / 512 通道和 1x1 层上普遍更慢，所以共享内存充足的卡默认 1，其余默认 0。
// TT_TC_CFG=small / auto / big 可强制指定
int BigTiles(){
    static int big = [](){
        const char* e = getenv("TT_TC_CFG");
        if (e && std::string(e) == "small") return 0;
        if (e && std::string(e) == "auto") return 1;
        if (e && std::string(e) == "big") return 2;
        return Stages() == 4 ? 1 : 0;
    }();
    return big;
}

template <int MODE>
void Dispatch(const bf16* a, const bf16* b, void* out, TcArgs p){
    const int bn = p.gn <= 64 ? 64 : 128;
    const int bm = MODE == WGRAD && p.gm <= 64 ? 64 : 128;
    if (MODE == WGRAD){
        // 输出（K x RSC）通常很小而像素维很长：split-K 凑够约 1.5 波 block（每个 SM 2 个 block）
        int tiles = ((p.gm + bm - 1) / bm) * ((p.gn + bn - 1) / bn) * p.g.groups;
        int want = std::max(1, 3 * DeviceSmCount() / tiles);
        int max_splits = std::max(1, p.gk / (BK * 8));
        p.splits = std::min(want, max_splits);
        p.k_per_split = ((p.gk + p.splits - 1) / p.splits + BK - 1) / BK * BK;
        p.splits = (p.gk + p.k_per_split - 1) / p.k_per_split;
    }
    const int big = BigTiles();
    if (big){
        if (MODE == WGRAD && bm == 64){ LaunchStages<MODE, 64, 128, 2, 2>(a, b, out, p); return; }
        if (bn == 64 && MODE != WGRAD){ LaunchStages<MODE, 256, 64, 4, 1>(a, b, out, p); return; }
        if (big == 2){ LaunchStages<MODE, 128, 128, 2, 2>(a, b, out, p); return; }
    }
    if (MODE == WGRAD && bm == 64){
        if (bn == 64) LaunchStages<MODE, 64, 64>(a, b, out, p);
        else LaunchStages<MODE, 64, 128>(a, b, out, p);
    }
    else if (bn == 64) LaunchStages<MODE, 128, 64>(a, b, out, p);
    else LaunchStages<MODE, 128, 128>(a, b, out, p);
}

}  // namespace

bool ConvTensorCoreSupported(const ConvGeom& g, int mode){
    const int cg = g.c / g.groups, kg = g.k / g.groups;
    if (mode == FWD) return cg % 32 == 0;
    if (mode == DGRAD) return kg % 32 == 0 && cg % 8 == 0 && g.h % g.stride_h == 0 && g.w % g.stride_w == 0;
    return kg % 8 == 0 && cg % 8 == 0;
}

void ConvTensorCore(int mode, const bf16* a, const bf16* b, void* out, const ConvGeom& g){
    TcArgs p = MakeTcArgs(g, mode);
    if (mode == FWD) Dispatch<FWD>(a, b, out, p);
    else if (mode == DGRAD) Dispatch<DGRAD>(a, b, out, p);
    else Dispatch<WGRAD>(a, b, out, p);
}
