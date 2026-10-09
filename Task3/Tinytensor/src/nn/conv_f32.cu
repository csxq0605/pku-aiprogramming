// fp32 卷积（真正的 fp32 FMA，不用 TF32）：隐式 GEMM，128 x BN x 8 的 tile，每个线程算 8 x (BN/16) 个输出
// - 全局内存按 float4 读（通道数需对齐到 4 / 8），先读进寄存器，算当前 tile 的同时预取下一个 tile（双缓冲）
// - 共享内存按 [k][m] / [k][n] 存放，计算时每个线程用 float4 读出连续 4 行 / 4 列
// GEMM 的定义与 conv.cu / conv_tc.cu 相同；不满足对齐要求时由 conv.cu 回退到通用 SIMT 版本
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

enum Mode { FWD = 0, DGRAD = 1, WGRAD = 2 };
constexpr int BM = 128, BK = 8, THREADS = 256, PAD = 4;

struct F32Args{
    ConvGeom g;
    int cg, kg, gm, gn, gk, k_per_split, splits;
    int bpp;             // DGRAD：每个相位的 block 数（与 conv_tc.cu 相同的按步长相位分组）
};

// A / B 在全局内存中是否沿 K 连续：是则读入后转置存进共享内存
template <int MODE> struct Lay{ static constexpr bool A_K = MODE != WGRAD; static constexpr bool B_K = MODE == FWD; };

template <int MODE, int BN>
__global__ void __launch_bounds__(THREADS) ConvF32(const float* __restrict__ a, const float* __restrict__ b,
                                                   float* __restrict__ out, const F32Args p){
    __shared__ __align__(16) float As[2][BK][BM + PAD];
    __shared__ __align__(16) float Bs[2][BK][BN + PAD];
    constexpr bool A_K = Lay<MODE>::A_K, B_K = Lay<MODE>::B_K;
    constexpr int TN = BN / 16;                      // 每个线程的列数：8 或 4
    const ConvGeom& g = p.g;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int group = blockIdx.z / p.splits, split = blockIdx.z % p.splits;
    const int k_begin = split * p.k_per_split;
    const int k_end = min(p.gk, k_begin + p.k_per_split);
    int ktiles = (k_end - k_begin + BK - 1) / BK;
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

    // A 沿 K 连续时（FWD / DGRAD）：线程 t 负责第 t/2 行的 4 个 k
    const int a_row = tid >> 1, a_kq = (tid & 1) * 4;
    int an = 0, ah = 0, aw = 0;
    bool aok = false;
    if (A_K){
        int m = m0 + a_row;
        aok = m < p.gm;
        int mm = aok ? m : 0;
        if (MODE == FWD){
            int ow = mm % g.q; int t = mm / g.q; int oh = t % g.p;
            an = t / g.p; ah = oh * g.stride_h - g.pad_h; aw = ow * g.stride_w - g.pad_w;
        }
        else{
            int iw = mm % Wp; int t = mm / Wp; int ih = t % Hp;
            an = t / Hp; ah = ih * g.stride_h + ph_h + g.pad_h; aw = iw * g.stride_w + ph_w + g.pad_w;
        }
    }

    float4 ra, rb;
    const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    auto load = [&](int kt){
        const int k0 = k_begin + kt * BK;
        int tap_r = 0, tap_s = 0, tap_c = 0;
        if (MODE == DGRAD){
            const int tpt = p.kg / BK, ti = kt / tpt;
            tap_c = (kt % tpt) * BK;
            tap_r = r0 + (ti / ns) * g.stride_h; tap_s = s0 + (ti % ns) * g.stride_w;
        }
        // ---- A ----
        ra = zero;
        if (A_K){
            const int cb = MODE == FWD ? k0 % p.cg : tap_c;
            const int rs = k0 / p.cg;
            const int s = MODE == FWD ? rs % g.s : tap_s, r = MODE == FWD ? rs / g.s : tap_r;
            if (MODE == FWD){
                int ih = ah + r, iw = aw + s;
                if (aok && ih >= 0 && ih < g.h && iw >= 0 && iw < g.w)
                    ra = __ldg((const float4*)(a + (((size_t)an * g.h + ih) * g.w + iw) * g.c + group * p.cg + cb + a_kq));
            }
            else{
                int th = ah - r, tw = aw - s;
                if (aok && th >= 0 && tw >= 0){
                    int oh = th / g.stride_h, ow = tw / g.stride_w;
                    if (oh < g.p && ow < g.q)
                        ra = __ldg((const float4*)(a + (((size_t)an * g.p + oh) * g.q + ow) * g.k + group * p.kg + cb + a_kq));
                }
            }
        }
        else{
            // WGRAD：A[k = 像素][m = 输出通道]，线程 t 负责第 t/32 行的 4 个 m
            int pix = k0 + (tid >> 5), m = m0 + (tid & 31) * 4;
            if (pix < k_end && m < p.gm) ra = __ldg((const float4*)(a + (size_t)pix * g.k + group * p.kg + m));
        }
        // ---- B ----
        rb = zero;
        if (B_K){
            // FWD：B[n][k] = w[group*kg + n][k]
            int row = tid >> 1, n = n0 + row;
            if (row < BN && n < p.gn) rb = __ldg((const float4*)(b + (size_t)(group * p.kg + n) * p.gk + k0 + (tid & 1) * 4));
        }
        else if (tid < BK * BN / 4){
            constexpr int CPR = BN / 4;
            int krow = tid / CPR, n = n0 + (tid % CPR) * 4, kk = k0 + krow;
            if (n < p.gn){
                if (MODE == DGRAD){
                    int kch = tap_c + krow;
                    rb = __ldg((const float4*)(b + (((size_t)(group * p.kg + kch) * g.r + tap_r) * g.s + tap_s) * p.cg + n));
                }
                else if (kk < k_end){
                    int ow = kk % g.q; int t = kk / g.q; int oh = t % g.p; int ni = t / g.p;
                    int c = n % p.cg; t = n / p.cg; int s = t % g.s, r = t / g.s;
                    int ih = oh * g.stride_h - g.pad_h + r, iw = ow * g.stride_w - g.pad_w + s;
                    if (ih >= 0 && ih < g.h && iw >= 0 && iw < g.w)
                        rb = __ldg((const float4*)(b + (((size_t)ni * g.h + ih) * g.w + iw) * g.c + group * p.cg + c));
                }
            }
        }
    };
    auto store = [&](int buf){
        if (A_K){
            As[buf][a_kq + 0][a_row] = ra.x; As[buf][a_kq + 1][a_row] = ra.y;
            As[buf][a_kq + 2][a_row] = ra.z; As[buf][a_kq + 3][a_row] = ra.w;
        }
        else{
            *(float4*)&As[buf][tid >> 5][(tid & 31) * 4] = ra;
        }
        if (B_K){
            int row = tid >> 1, kq = (tid & 1) * 4;
            if (row < BN){
                Bs[buf][kq + 0][row] = rb.x; Bs[buf][kq + 1][row] = rb.y;
                Bs[buf][kq + 2][row] = rb.z; Bs[buf][kq + 3][row] = rb.w;
            }
        }
        else if (tid < BK * BN / 4){
            constexpr int CPR = BN / 4;
            *(float4*)&Bs[buf][tid / CPR][(tid % CPR) * 4] = rb;
        }
    };

    float acc[8][TN];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    if (ktiles > 0) load(0);
    else{ ra = zero; rb = zero; }
    store(0);
    __syncthreads();
    int buf = 0;
    for (int kt = 0; kt < ktiles; ++kt){
        const bool more = kt + 1 < ktiles;
        if (more) load(kt + 1);
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk){
            float av[8], bv[TN];
            float4 t0 = *(const float4*)&As[buf][kk][ty * 4];
            float4 t1 = *(const float4*)&As[buf][kk][64 + ty * 4];
            av[0] = t0.x; av[1] = t0.y; av[2] = t0.z; av[3] = t0.w;
            av[4] = t1.x; av[5] = t1.y; av[6] = t1.z; av[7] = t1.w;
            float4 u0 = *(const float4*)&Bs[buf][kk][tx * 4];
            bv[0] = u0.x; bv[1] = u0.y; bv[2] = u0.z; bv[3] = u0.w;
            if (TN == 8){
                float4 u1 = *(const float4*)&Bs[buf][kk][64 + tx * 4];
                bv[TN > 4 ? 4 : 0] = u1.x; bv[TN > 4 ? 5 : 1] = u1.y; bv[TN > 4 ? 6 : 2] = u1.z; bv[TN > 4 ? 7 : 3] = u1.w;
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        if (more){
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i){
        int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (m >= p.gm) continue;
        size_t row = m;
        if (MODE == DGRAD){
            int iw = m % Wp; int t = m / Wp; int ih = t % Hp; int n = t / Hp;
            row = ((size_t)n * g.h + ih * g.stride_h + ph_h) * g.w + iw * g.stride_w + ph_w;
        }
        #pragma unroll
        for (int jh = 0; jh < TN / 4; ++jh){
            int n = n0 + jh * 64 + tx * 4;
            const float* v = &acc[i][jh * 4];
            if (MODE == WGRAD){
                float* dst = out + (size_t)(group * p.kg + m) * p.gn + n;
                #pragma unroll
                for (int e = 0; e < 4; ++e) if (n + e < p.gn) atomicAdd(dst + e, v[e]);
            }
            else{
                size_t ld = MODE == FWD ? g.k : g.c;
                int col = (MODE == FWD ? group * p.kg : group * p.cg) + n;
                float* dst = out + row * ld + col;
                if (n + 3 < p.gn && ((row * ld + col) & 3) == 0) *(float4*)dst = make_float4(v[0], v[1], v[2], v[3]);
                else{
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) if (n + e < p.gn) dst[e] = v[e];
                }
            }
        }
    }
}

template <int MODE>
void Dispatch(const float* a, const float* b, float* out, const ConvGeom& g){
    F32Args p;
    p.g = g; p.cg = g.c / g.groups; p.kg = g.k / g.groups;
    if (MODE == FWD){ p.gm = g.n * g.p * g.q; p.gn = p.kg; p.gk = g.r * g.s * p.cg; }
    else if (MODE == DGRAD){ p.gm = g.n * (g.h / g.stride_h) * (g.w / g.stride_w); p.gn = p.cg; p.gk = g.r * g.s * p.kg; }
    else { p.gm = p.kg; p.gn = g.r * g.s * p.cg; p.gk = g.n * g.p * g.q; }
    const int bn = p.gn <= 64 ? 64 : 128;
    p.splits = 1; p.k_per_split = p.gk;
    if (MODE == WGRAD){
        int tiles = ((p.gm + BM - 1) / BM) * ((p.gn + bn - 1) / bn) * g.groups;
        int want = std::max(1, 96 / tiles);
        int max_splits = std::max(1, p.gk / (BK * 32));
        p.splits = std::min(want, max_splits);
        p.k_per_split = ((p.gk + p.splits - 1) / p.splits + BK - 1) / BK * BK;
        p.splits = (p.gk + p.k_per_split - 1) / p.k_per_split;
    }
    p.bpp = (p.gm + BM - 1) / BM;
    int gx = p.bpp * (MODE == DGRAD ? g.stride_h * g.stride_w : 1);
    dim3 grid(gx, (p.gn + bn - 1) / bn, g.groups * p.splits);
    if (bn == 64) ConvF32<MODE, 64><<<grid, THREADS>>>(a, b, out, p);
    else ConvF32<MODE, 128><<<grid, THREADS>>>(a, b, out, p);
}

}  // namespace

bool ConvF32Supported(const ConvGeom& g, int mode){
    const int cg = g.c / g.groups, kg = g.k / g.groups;
    if (mode == FWD) return cg % 8 == 0;
    if (mode == DGRAD) return kg % 8 == 0 && cg % 4 == 0 && g.h % g.stride_h == 0 && g.w % g.stride_w == 0;
    return kg % 4 == 0 && cg % 4 == 0;
}

void ConvF32(int mode, const float* a, const float* b, float* out, const ConvGeom& g){
    if (mode == FWD) Dispatch<FWD>(a, b, out, g);
    else if (mode == DGRAD) Dispatch<DGRAD>(a, b, out, g);
    else Dispatch<WGRAD>(a, b, out, g);
}
