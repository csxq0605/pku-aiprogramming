// 每组通道很少的分组卷积（ResNeXt 的 3x3，每组 4 / 8 / 16 个通道）：直接卷积，不走 GEMM
// 这类卷积的计算量很小（每个输出只有 R*S*CG 次乘加），瓶颈在访存；用 GEMM 做会浪费大部分 tile。
// 要求：输入输出通道数相同（C == K）、每组输入输出通道数相同（cg == kg = CG）、C 是 64 的倍数
// 一个 block 负责 64 个通道（及其权重，整块放进共享内存），block 常驻、依次处理多段像素：
//   前向 / 输入梯度：warp w 负责其中 8 个通道，lane 对应像素；权重在 warp 内是统一地址（共享内存广播）
//   权重梯度：dy 与输入窗口先读进共享内存，线程 = (通道, 抽头组) 沿像素累加，最后 atomicAdd
#include <cstdlib>
#include <type_traits>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

constexpr int SLICE = 64, THREADS = 256, PIX_PER_THREAD = 4;

// 从 p 读 N 个连续元素（N 为 8 的倍数，p 按 16 字节对齐）
template <int N>
__device__ __forceinline__ void LoadVec(const float* p, float* v){
    #pragma unroll
    for (int i = 0; i < N / 4; ++i){
        float4 t = __ldg((const float4*)p + i);
        v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
    }
}
template <int N>
__device__ __forceinline__ void LoadVec(const bf16* p, float* v){
    #pragma unroll
    for (int i = 0; i < N / 8; ++i){
        uint4 t = __ldg((const uint4*)p + i);
        const __nv_bfloat162* h = (const __nv_bfloat162*)&t;
        #pragma unroll
        for (int j = 0; j < 4; ++j){
            float2 f = __bfloat1622float2(h[j]);
            v[8 * i + 2 * j] = f.x; v[8 * i + 2 * j + 1] = f.y;
        }
    }
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

// 把本 block 的 64 个通道的权重放进共享内存，排成 ws[jl][tap][q]（jl = 切片内的输出通道，q = 组内的另一维）：
//   前向：ws[jl][tap][q] = w[jl][tap][q]（在全局内存里本来就是这个顺序）
//   DGRAD：jl 是输入通道，q 是同组的输出通道：ws[jl][tap][q] = w[jl/CG*CG + q][tap][jl%CG]
// 这样两种模式内层都是沿 q 连续读，可以 float4 一次读 4 个
template <typename T, int CG, bool DGRAD>
__device__ __forceinline__ void LoadWeights(float* ws, const T* w, int slice, int taps){
    const T* src = w + (size_t)slice * SLICE * taps * CG;
    const int n = SLICE * taps * CG;
    for (int i = threadIdx.x; i < n; i += blockDim.x){
        if (!DGRAD) ws[i] = ToF(src[i]);
        else{
            // i 是源下标：[kl][tap][c]
            int c = i % CG; int t = i / CG; int tap = t % taps; int kl = t / taps;
            int jl = kl / CG * CG + c, q = kl % CG;
            ws[(jl * taps + tap) * CG + q] = ToF(src[i]);
        }
    }
}

// 前向：y[p][k] = sum_{r,s,ci} x[p 的 (r,s) 抽头][grp*CG + ci] * w[k][r][s][ci]
// DGRAD=true 时同一个 kernel 算输入梯度：dx[p][c] = sum_{r,s,kk} dy[对应的输出像素][grp*CG + kk] * w[grp*CG + kk][r][s][c 在组内的下标]
// block 常驻：权重只读一次，然后依次处理若干段 32 * PIX_PER_THREAD 个像素
template <typename T, int CG, bool DGRAD>
__global__ void __launch_bounds__(THREADS) GroupedDirect(const T* __restrict__ in, const T* __restrict__ w,
                                                         T* __restrict__ out, const ConvGeom g){
    extern __shared__ float ws[];                       // [SLICE][R*S][CG]
    constexpr int IN = CG > 8 ? CG : 8;                 // 每个线程读入的通道数
    const int taps = g.r * g.s;
    const int slice = blockIdx.y;
    LoadWeights<T, CG, DGRAD>(ws, w, slice, taps);
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = slice * SLICE + warp * 8;            // 本 warp 的 8 个输出通道（DGRAD 时是输入通道）
    const int cin0 = CG > 8 ? c0 / CG * CG : c0;        // 需要读的输入通道起点
    const int local0 = warp * 8;                        // 在 64 通道切片内的下标
    // 输出像素数（DGRAD 时是输入像素数）
    const int out_h = DGRAD ? g.h : g.p, out_w = DGRAD ? g.w : g.q;
    const int in_h = DGRAD ? g.p : g.h, in_w = DGRAD ? g.q : g.w;
    const int in_c = DGRAD ? g.k : g.c, out_c = DGRAD ? g.c : g.k;
    const int total = g.n * out_h * out_w;
    const int chunks = (total + 32 * PIX_PER_THREAD - 1) / (32 * PIX_PER_THREAD);

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x){
        #pragma unroll 1
        for (int it = 0; it < PIX_PER_THREAD; ++it){
            const int m = (chunk * PIX_PER_THREAD + it) * 32 + lane;
            if (m >= total) break;
            int ow = m % out_w; int t = m / out_w; int oh = t % out_h; int n = t / out_h;
            float acc[8] = {};
            for (int r = 0; r < g.r; ++r){
                for (int s = 0; s < g.s; ++s){
                    int ih, iw;
                    if (!DGRAD){
                        ih = oh * g.stride_h - g.pad_h + r; iw = ow * g.stride_w - g.pad_w + s;
                    }
                    else{
                        int th = oh + g.pad_h - r, tw = ow + g.pad_w - s;
                        if (th < 0 || tw < 0 || th % g.stride_h || tw % g.stride_w) continue;
                        ih = th / g.stride_h; iw = tw / g.stride_w;
                    }
                    if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                    float v[IN];
                    LoadVec<IN>(in + (((size_t)n * in_h + ih) * in_w + iw) * in_c + cin0, v);
                    const float* wt = ws + (r * g.s + s) * CG;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j){
                        const int base = CG > 8 ? 0 : (j / CG) * CG;            // 通道 j 所在组在 v 中的起点
                        const float4* wj = (const float4*)(wt + (local0 + j) * taps * CG);
                        #pragma unroll
                        for (int q4 = 0; q4 < CG / 4; ++q4){
                            float4 u = wj[q4];                                   // warp 内统一地址：广播
                            acc[j] = fmaf(v[base + q4 * 4 + 0], u.x, acc[j]);
                            acc[j] = fmaf(v[base + q4 * 4 + 1], u.y, acc[j]);
                            acc[j] = fmaf(v[base + q4 * 4 + 2], u.z, acc[j]);
                            acc[j] = fmaf(v[base + q4 * 4 + 3], u.w, acc[j]);
                        }
                    }
                }
            }
            Store8(out + (size_t)m * out_c + c0, acc);
        }
    }
}

// 权重梯度：dw[k][r][s][q] += sum_p dy[p][k] * x[p 的 (r,s) 抽头][grp(k)*CG + q]
// block 负责一个 64 通道切片，常驻并依次处理若干个像素 tile（TR 行 x TQ 列个输出像素，TR*TQ <= 32）：
//   dy 的 tile [32][64] 与对应的输入窗口 [XR][XC][64] 先读进共享内存（fp32），
//   线程 = (切片内通道 kl, 抽头组 tg)：负责抽头 tg, tg+4, tg+8（3x3 时最多 3 个），每个抽头 CG 个累加器，
//   输入从共享内存按 float4 读（同组的 lane 读同一地址，广播）。最后 atomicAdd 到全局。
struct WgTile{ int tr, tq, xr, xc, tiles_h, tiles_w, tiles; };

template <typename T, int CG>
__global__ void __launch_bounds__(THREADS) GroupedWgrad(const T* __restrict__ x, const T* __restrict__ dy,
                                                        float* __restrict__ dw, const ConvGeom g, const WgTile tl){
    constexpr int TPT = 3;                               // 每个线程最多 3 个抽头（taps <= 12）
    constexpr int TPIX = 32;
    extern __shared__ float4 smem4[];
    float* dys = (float*)smem4;                          // [TPIX][SLICE]
    float* xs = dys + TPIX * SLICE;                      // [XR * XC][SLICE]
    const int kl = threadIdx.x % SLICE, tg = threadIdx.x / SLICE;
    const int slice = blockIdx.y, c0 = slice * SLICE;
    const int cl = kl / CG * CG;                         // 本通道所在组在切片内的输入通道起点
    const int taps = g.r * g.s;
    int toff[TPT];                                       // 抽头在输入窗口里的偏移（单位：格）
    #pragma unroll
    for (int a = 0; a < TPT; ++a){
        int tap = tg + 4 * a;
        toff[a] = tap < taps ? (tap / g.s) * tl.xc + tap % g.s : -1;
    }
    float acc[TPT][CG];
    #pragma unroll
    for (int a = 0; a < TPT; ++a)
        #pragma unroll
        for (int q = 0; q < CG; ++q) acc[a][q] = 0.f;

    for (int tile = blockIdx.x; tile < tl.tiles; tile += gridDim.x){
        const int tw = tile % tl.tiles_w; int t = tile / tl.tiles_w;
        const int th = t % tl.tiles_h, n = t / tl.tiles_h;
        const int oh0 = th * tl.tr, ow0 = tw * tl.tq;
        __syncthreads();
        // dy tile：越界像素填 0（于是不必再判断）
        for (int idx = threadIdx.x; idx < TPIX * SLICE / 8; idx += THREADS){
            int pix = idx / (SLICE / 8), c8 = (idx % (SLICE / 8)) * 8;
            int pr = pix / tl.tq, pc = pix % tl.tq;
            int oh = oh0 + pr, ow = ow0 + pc;
            float v[8] = {};
            if (pr < tl.tr && oh < g.p && ow < g.q) LoadVec<8>(dy + (((size_t)n * g.p + oh) * g.q + ow) * g.k + c0 + c8, v);
            float4* d = (float4*)(dys + pix * SLICE + c8);
            d[0] = make_float4(v[0], v[1], v[2], v[3]); d[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        // 输入窗口
        for (int idx = threadIdx.x; idx < tl.xr * tl.xc * (SLICE / 8); idx += THREADS){
            int cell = idx / (SLICE / 8), c8 = (idx % (SLICE / 8)) * 8;
            int xr = cell / tl.xc, xc = cell % tl.xc;
            int ih = oh0 * g.stride_h - g.pad_h + xr, iw = ow0 * g.stride_w - g.pad_w + xc;
            float v[8] = {};
            if (ih >= 0 && ih < g.h && iw >= 0 && iw < g.w) LoadVec<8>(x + (((size_t)n * g.h + ih) * g.w + iw) * g.c + c0 + c8, v);
            float4* d = (float4*)(xs + cell * SLICE + c8);
            d[0] = make_float4(v[0], v[1], v[2], v[3]); d[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
        __syncthreads();
        for (int pr = 0; pr < tl.tr; ++pr){
            for (int pc = 0; pc < tl.tq; ++pc){
                const float d = dys[(pr * tl.tq + pc) * SLICE + kl];
                const int base = pr * g.stride_h * tl.xc + pc * g.stride_w;
                #pragma unroll
                for (int a = 0; a < TPT; ++a){
                    if (toff[a] < 0) break;
                    const float4* v = (const float4*)(xs + (base + toff[a]) * SLICE + cl);
                    #pragma unroll
                    for (int q4 = 0; q4 < CG / 4; ++q4){
                        float4 u = v[q4];
                        acc[a][q4 * 4 + 0] = fmaf(d, u.x, acc[a][q4 * 4 + 0]);
                        acc[a][q4 * 4 + 1] = fmaf(d, u.y, acc[a][q4 * 4 + 1]);
                        acc[a][q4 * 4 + 2] = fmaf(d, u.z, acc[a][q4 * 4 + 2]);
                        acc[a][q4 * 4 + 3] = fmaf(d, u.w, acc[a][q4 * 4 + 3]);
                    }
                }
            }
        }
    }
    const int k = c0 + kl;
    #pragma unroll
    for (int a = 0; a < TPT; ++a){
        int tap = tg + 4 * a;
        if (tap >= taps) break;
        float* dst = dw + ((size_t)k * taps + tap) * CG;
        #pragma unroll
        for (int q = 0; q < CG; ++q) atomicAdd(dst + q, acc[a][q]);
    }
}

WgTile MakeWgTile(const ConvGeom& g){
    WgTile t;
    t.tq = std::min(g.q, 32);
    t.tr = std::max(1, 32 / t.tq);
    t.xr = (t.tr - 1) * g.stride_h + g.r;
    t.xc = (t.tq - 1) * g.stride_w + g.s;
    t.tiles_h = (g.p + t.tr - 1) / t.tr;
    t.tiles_w = (g.q + t.tq - 1) / t.tq;
    t.tiles = g.n * t.tiles_h * t.tiles_w;
    return t;
}

// ---------------- bf16 的 tensor core 版本（mma.sync.m16n8k16）----------------
// 把 16 个通道（一个 slab，含 16/CG 个完整的组）当成一个 16x16 的块对角 GEMM：CG = 16 时是满块，CG = 8 / 4 时非对角块填 0
// （浪费 1/2、3/4 的 MMA；这类卷积卡在访存上，tensor core 的算力绰绰有余）。
// block 负责 64 个通道（4 个 slab）和一块像素 tile（NI 张图 x TH 行 x TW 列，最多 128 个像素），
// 输入窗口（含 halo）整块读进共享内存一次，每个抽头的 A / B 直接用 ldmatrix 从窗口里按行收集：
// ldmatrix 每一行的地址由各 lane 自己给，所以 im2col 不需要展开。
//   FWD  : 行 = 输出像素，窗口在 x 上，窗口内采样步长 = 卷积步长，抽头 (i, j) = (r, s)
//   DGRAD: 行 = 输入像素，按步长分相位（同 conv_tc.cu），每个相位只循环有效抽头；窗口在 dy 上、采样步长 1，
//          第 i 个抽头对应 r = r0 + (nr - 1 - i) * sh
//   WGRAD: 每个 warp = (slab, 一半抽头)，沿 tile 内的像素做 K 维累加：A = dy tile 转置，B = 窗口里按像素收集的行；
//          常驻 block 遍历完所有 tile 后，只把块对角部分 atomicAdd 到 dw
// block 常驻（权重只读一次），依次处理多个 tile；每个 SM 驻留 2~3 个 block，一个读窗口时其它的在算
namespace gtc {

constexpr int PIX = 128, CS = 72;                    // 每个 tile 最多 128 个像素；窗口每格 64 通道 + 8 填充（144 字节，错开 bank）
constexpr int THREADS = 256;                         // 8 个 warp；FWD / DGRAD 每个 warp 16 行
constexpr int MAXC = 300;                            // 窗口最多格数（约 43KB）
constexpr int TPW = 6;                               // WGRAD 每个 warp 最多 6 个抽头（taps <= 12）

__device__ __forceinline__ unsigned Smem(const void* p){ return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void CpAsync16(void* dst, const void* src, bool valid){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(Smem(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void CpCommitWait(){ asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::); }
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
// 权重块 [16 n][16 k]，每行 2 个 16 字节块，块号与 (n >> 2) & 1 异或（ldmatrix 的 8 行落在不同 bank）
__device__ __forceinline__ int OffW(int n, int k){ return n * 16 + ((((k >> 3) ^ (n >> 2)) & 1) << 3) + (k & 7); }

// tile 划分：本算子每个相位的输出网格 OH x OW；窗口在采样步长 (ssh, ssw)、抽头数 (nr, ns) 下的大小 XR x XC
struct Tile{
    int th, tw, ni, xr, xc, cells;
    int tiles_h, tiles_w, per_phase, total;
};

Tile MakeTile(int oh, int ow, int n, int ssh, int ssw, int nr, int ns, int phases){
    Tile t;
    t.tw = std::min(ow, 16); t.th = std::min(oh, PIX / t.tw); t.ni = 1;
    if (t.th == oh && t.tw == ow) t.ni = std::max(1, std::min(n, PIX / (oh * ow)));
    for (;;){
        t.xr = (t.th - 1) * ssh + nr; t.xc = (t.tw - 1) * ssw + ns;
        t.cells = t.ni * t.xr * t.xc;
        if (t.cells <= MAXC) break;
        if (t.ni > 1) t.ni = (t.ni + 1) / 2;
        else if (t.th > 1) t.th = (t.th + 1) / 2;
        else t.tw = (t.tw + 1) / 2;
    }
    t.tiles_h = (oh + t.th - 1) / t.th; t.tiles_w = (ow + t.tw - 1) / t.tw;
    t.per_phase = (n + t.ni - 1) / t.ni * t.tiles_h * t.tiles_w;
    t.total = t.per_phase * phases;
    return t;
}

// 一个 tile 的位置：相位、起始图 / 行 / 列，相位决定的抽头范围，窗口原点（在窗口所在张量上的坐标）
struct TilePos{ int ph_h, ph_w, n0, y0, x0, nr, ns, r0, s0, org_h, org_w; };

template <int MODE>
__device__ __forceinline__ TilePos Locate(int tile, const Tile& t, const ConvGeom& g){
    TilePos q;
    const int phase = tile / t.per_phase; int rem = tile - phase * t.per_phase;
    const int tx = rem % t.tiles_w; rem /= t.tiles_w;
    const int ty = rem % t.tiles_h; const int ig = rem / t.tiles_h;
    q.n0 = ig * t.ni; q.y0 = ty * t.th; q.x0 = tx * t.tw;
    if (MODE == 1){
        q.ph_h = phase / g.stride_w; q.ph_w = phase % g.stride_w;
        q.r0 = (q.ph_h + g.pad_h) % g.stride_h; q.s0 = (q.ph_w + g.pad_w) % g.stride_w;
        q.nr = q.r0 < g.r ? (g.r - q.r0 + g.stride_h - 1) / g.stride_h : 0;
        q.ns = q.s0 < g.s ? (g.s - q.s0 + g.stride_w - 1) / g.stride_w : 0;
        // 相位内第 ih' 行的输入像素，第 i 个有效抽头读 dy 的 oh = ih' + ch - (nr - 1) + i，ch = (ph_h + pad_h - r0) / sh
        q.org_h = q.y0 + (q.ph_h + g.pad_h - q.r0) / g.stride_h - (q.nr - 1);
        q.org_w = q.x0 + (q.ph_w + g.pad_w - q.s0) / g.stride_w - (q.ns - 1);
    }
    else{
        q.ph_h = q.ph_w = 0; q.r0 = q.s0 = 0; q.nr = g.r; q.ns = g.s;
        q.org_h = q.y0 * g.stride_h - g.pad_h; q.org_w = q.x0 * g.stride_w - g.pad_w;
    }
    return q;
}

// 窗口：NI x XR x XC 格，每格本 block 的 64 个通道；src 是 [N][ih][iw][C] 的张量，越界的格由 cp.async 填 0
__device__ __forceinline__ void LoadWindow(bf16* win, const bf16* src, int ih, int iw, int nimg, int ldc, int c0,
                                           const Tile& t, const TilePos& q){
    for (int id = threadIdx.x; id < t.cells * 8; id += THREADS){
        const int cell = id >> 3, ch = (id & 7) * 8;
        const int xc = cell % t.xc; const int r = cell / t.xc; const int xr = r % t.xr; const int img = r / t.xr;
        const int n = q.n0 + img, h = q.org_h + xr, w = q.org_w + xc;
        const bool ok = n < nimg && h >= 0 && h < ih && w >= 0 && w < iw;
        const bf16* p = ok ? src + (((size_t)n * ih + h) * iw + w) * ldc + c0 + ch : src;
        CpAsync16(win + cell * CS + ch, p, ok);
    }
}

// tile 内第 pix 个像素（行）：窗口里的起始格（抽头 (0,0)），以及本算子输出里的像素下标（无效为 -1）
// ssh / ssw：窗口内的采样步长；oh / ow：每个相位的输出网格；out_*：真正的输出张量（DGRAD 时按相位映射回去）
template <int MODE>
__device__ __forceinline__ void PixelTable(int* pcell, int* pout, const Tile& t, const TilePos& q, const ConvGeom& g,
                                           int oh, int ow, int ssh, int ssw){
    for (int pix = threadIdx.x; pix < PIX; pix += THREADS){
        const int pc = pix % t.tw; const int r = pix / t.tw; const int pr = r % t.th; const int img = r / t.th;
        const int n = q.n0 + img, y = q.y0 + pr, x = q.x0 + pc;
        const bool ok = img < t.ni && n < g.n && y < oh && x < ow;
        pcell[pix] = ok ? (img * t.xr + pr * ssh) * t.xc + pc * ssw : 0;
        int o = -1;
        if (ok){
            if (MODE == 1) o = (n * g.h + y * g.stride_h + q.ph_h) * g.w + x * g.stride_w + q.ph_w;
            else o = (n * oh + y) * ow + x;
        }
        pout[pix] = o;
    }
}

// 权重读进共享内存：ws[tap][slab][16 n][16 k]（tap = r * S + s，块对角之外填 0）
//   FWD  ：n = 输出通道，k = 输入通道，值 w[n][r][s][k % CG]
//   DGRAD：n = 输入通道，k = 输出通道，值 w[k][r][s][n % CG]
template <int CG, int MODE>
__device__ __forceinline__ void LoadWeightsTc(bf16* ws, const bf16* w, int c0, int taps){
    for (int i = threadIdx.x; i < taps * 4 * 256; i += THREADS){
        const int k = i & 15, n = (i >> 4) & 15, slab = (i >> 8) & 3, tap = i >> 10;
        bf16 v = __float2bfloat16(0.f);
        if (n / CG == k / CG){
            const int cn = c0 + slab * 16 + n, ck = c0 + slab * 16 + k;
            v = MODE == 0 ? w[((size_t)cn * taps + tap) * CG + ck % CG] : w[((size_t)ck * taps + tap) * CG + cn % CG];
        }
        ws[(tap * 4 + slab) * 256 + OffW(n, k)] = v;
    }
}

// FWD / DGRAD
template <int CG, int MODE>
__global__ void __launch_bounds__(THREADS) GroupedTc(const bf16* __restrict__ in, const bf16* __restrict__ w,
                                                     bf16* __restrict__ out, const ConvGeom g, const Tile t){
    extern __shared__ __align__(128) unsigned char smem_raw[];
    const int taps = g.r * g.s;
    bf16* ws = (bf16*)smem_raw;                          // [taps][4][256]
    bf16* win = ws + taps * 4 * 256;                     // [cells][CS]
    int* pcell = (int*)(win + t.cells * CS);             // [PIX]
    int* pout = pcell + PIX;                             // [PIX]
    const int c0 = blockIdx.y * 64;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ih = MODE == 0 ? g.h : g.p, iw = MODE == 0 ? g.w : g.q;
    const int oh = MODE == 0 ? g.p : g.h / g.stride_h, ow = MODE == 0 ? g.q : g.w / g.stride_w;
    const int ssh = MODE == 0 ? g.stride_h : 1, ssw = MODE == 0 ? g.stride_w : 1;
    LoadWeightsTc<CG, MODE>(ws, w, c0, taps);

    // B 的 ldmatrix 偏移（每个 lane 固定）：r0/r1 = n 0~7，r2/r3 = n 8~15
    const int boff = OffW((lane & 7) + ((lane >> 4) << 3), ((lane >> 3) & 1) << 3);
    const int arow = warp * 16 + (lane & 15), acol = (lane >> 4) << 3;

    for (int tile = blockIdx.x; tile < t.total; tile += gridDim.x){
        const TilePos q = Locate<MODE>(tile, t, g);
        __syncthreads();                                 // 上一个 tile 的窗口 / 像素表已用完
        LoadWindow(win, in, ih, iw, g.n, g.c, c0, t, q);
        PixelTable<MODE>(pcell, pout, t, q, g, oh, ow, ssh, ssw);
        CpCommitWait();
        __syncthreads();

        float acc[4][2][4];
        #pragma unroll
        for (int s = 0; s < 4; ++s)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int e = 0; e < 4; ++e) acc[s][j][e] = 0.f;
        if (warp * 16 < t.ni * t.th * t.tw){
            const bf16* abase = win + pcell[arow] * CS + acol;
            for (int i = 0; i < q.nr; ++i){
                for (int j = 0; j < q.ns; ++j){
                    const int tap = MODE == 0 ? i * g.s + j
                                              : (q.r0 + (q.nr - 1 - i) * g.stride_h) * g.s + q.s0 + (q.ns - 1 - j) * g.stride_w;
                    const bf16* ap = abase + (i * t.xc + j) * CS;
                    const bf16* bp = ws + tap * 4 * 256 + boff;
                    #pragma unroll
                    for (int s = 0; s < 4; ++s){
                        unsigned a[4], b[4];
                        Ldm4<false>(a, ap + s * 16);
                        Ldm4<false>(b, bp + s * 256);
                        Mma(acc[s][0], a, b[0], b[1]);
                        Mma(acc[s][1], a, b[2], b[3]);
                    }
                }
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half){
                const int o = pout[warp * 16 + (lane >> 2) + half * 8];
                if (o < 0) continue;
                bf16* op = out + (size_t)o * g.c + c0 + (lane & 3) * 2;
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    #pragma unroll
                    for (int j = 0; j < 2; ++j)
                        *(__nv_bfloat162*)(op + s * 16 + j * 8) = __floats2bfloat162_rn(acc[s][j][half * 2], acc[s][j][half * 2 + 1]);
            }
        }
    }
}

// WGRAD：dw[k][r][s][q] += sum_pix dy[pix][k] * x[pix 的 (r,s) 抽头][grp(k)*CG + q]
template <int CG>
__global__ void __launch_bounds__(THREADS) GroupedWgradTc(const bf16* __restrict__ x, const bf16* __restrict__ dy,
                                                          float* __restrict__ dw, const ConvGeom g, const Tile t){
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* dys = (bf16*)smem_raw;                         // [PIX][CS]
    bf16* win = dys + PIX * CS;                          // [cells][CS]
    int* pcell = (int*)(win + t.cells * CS);
    int* pout = pcell + PIX;
    const int c0 = blockIdx.y * 64;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int slab = warp & 3, half = warp >> 2;
    const int taps = g.r * g.s;
    int toff[TPW];
    #pragma unroll
    for (int a = 0; a < TPW; ++a){
        const int tap = half + 2 * a;
        toff[a] = tap < taps ? (tap / g.s) * t.xc + tap % g.s : -1;
    }
    float acc[TPW][2][4];
    #pragma unroll
    for (int a = 0; a < TPW; ++a)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[a][j][e] = 0.f;
    const int npix = t.ni * t.th * t.tw;
    const int ksteps = (npix + 15) / 16;
    // A（dy^T）：lane 给出第 (lane & 7) + (lane >> 4) * 8 个像素行、第 ((lane >> 3) & 1) * 8 个通道起的 8 个通道
    const int a_pix = (lane & 7) + ((lane >> 4) << 3), a_ch = slab * 16 + (((lane >> 3) & 1) << 3);
    // B（窗口）：第 (lane & 7) + ((lane >> 3) & 1) * 8 个像素行、第 (lane >> 4) * 8 个通道起
    const int b_pix = (lane & 7) + (((lane >> 3) & 1) << 3), b_ch = slab * 16 + ((lane >> 4) << 3);

    for (int tile = blockIdx.x; tile < t.total; tile += gridDim.x){
        const TilePos q = Locate<0>(tile, t, g);
        __syncthreads();
        LoadWindow(win, x, g.h, g.w, g.n, g.c, c0, t, q);
        PixelTable<0>(pcell, pout, t, q, g, g.p, g.q, g.stride_h, g.stride_w);
        __syncthreads();
        // dy tile：无效像素填 0（于是窗口那边不用管）
        for (int id = threadIdx.x; id < PIX * 8; id += THREADS){
            const int pix = id >> 3, ch = (id & 7) * 8;
            const int o = pout[pix];
            CpAsync16(dys + pix * CS + ch, o >= 0 ? dy + (size_t)o * g.k + c0 + ch : dy, o >= 0);
        }
        CpCommitWait();
        __syncthreads();
        for (int ks = 0; ks < ksteps; ++ks){
            unsigned a[4];
            Ldm4<true>(a, dys + (ks * 16 + a_pix) * CS + a_ch);
            const bf16* bbase = win + pcell[ks * 16 + b_pix] * CS + b_ch;
            #pragma unroll
            for (int i = 0; i < TPW; ++i){
                if (toff[i] < 0) break;
                unsigned b[4];
                Ldm4<true>(b, bbase + toff[i] * CS);
                Mma(acc[i][0], a, b[0], b[1]);
                Mma(acc[i][1], a, b[2], b[3]);
            }
        }
    }
    // 只取块对角部分：行 m = 输出通道（slab 内），列 n = 同组的输入通道
    #pragma unroll
    for (int i = 0; i < TPW; ++i){
        const int tap = half + 2 * i;
        if (tap >= taps) break;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e){
                const int m = (lane >> 2) + (e >> 1) * 8, n = j * 8 + (lane & 3) * 2 + (e & 1);
                if (m / CG != n / CG) continue;
                const int k = c0 + slab * 16 + m;
                atomicAdd(dw + ((size_t)k * taps + tap) * CG + n % CG, acc[i][j][e]);
            }
    }
}

template <int CG>
void Run(int mode, const bf16* a, const bf16* b, void* out, const ConvGeom& g){
    const int slices = g.c / 64;
    const int taps = g.r * g.s;
    Tile t;
    int smem;
    const void* fn;
    if (mode == 0){
        t = MakeTile(g.p, g.q, g.n, g.stride_h, g.stride_w, g.r, g.s, 1);
        smem = taps * 4 * 256 * 2 + t.cells * CS * 2 + 2 * PIX * 4;
        fn = (const void*)GroupedTc<CG, 0>;
    }
    else if (mode == 1){
        const int nr = (g.r + g.stride_h - 1) / g.stride_h, ns = (g.s + g.stride_w - 1) / g.stride_w;
        t = MakeTile(g.h / g.stride_h, g.w / g.stride_w, g.n, 1, 1, nr, ns, g.stride_h * g.stride_w);
        smem = taps * 4 * 256 * 2 + t.cells * CS * 2 + 2 * PIX * 4;
        fn = (const void*)GroupedTc<CG, 1>;
    }
    else{
        t = MakeTile(g.p, g.q, g.n, g.stride_h, g.stride_w, g.r, g.s, 1);
        smem = PIX * CS * 2 + t.cells * CS * 2 + 2 * PIX * 4;
        fn = (const void*)GroupedWgradTc<CG>;
    }
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, THREADS, smem);
    const int bx = std::max(1, std::min(t.total, (DeviceSmCount() * std::max(1, per_sm) + slices - 1) / slices));
    dim3 grid(bx, slices);
    if (mode == 0) GroupedTc<CG, 0><<<grid, THREADS, smem>>>(a, b, (bf16*)out, g, t);
    else if (mode == 1) GroupedTc<CG, 1><<<grid, THREADS, smem>>>(a, b, (bf16*)out, g, t);
    else GroupedWgradTc<CG><<<grid, THREADS, smem>>>(b, a, (float*)out, g, t);
}

// TT_GROUPED_TC=0 关闭（对照实验用）；DGRAD 要求输入高宽是步长的整数倍（按相位划分）
bool Usable(int mode, const ConvGeom& g){
    static const bool on = [](){ const char* e = getenv("TT_GROUPED_TC"); return !(e && atoi(e) == 0); }();
    if (!on || g.r * g.s > 2 * TPW) return false;
    if (mode == 1) return g.h % g.stride_h == 0 && g.w % g.stride_w == 0;
    return true;
}

}  // namespace gtc

template <typename T, int CG>
void Run(int mode, const T* a, const T* b, void* out, const ConvGeom& g){
    if constexpr (std::is_same<T, bf16>::value){
        if (gtc::Usable(mode, g)){ gtc::Run<CG>(mode, a, b, out, g); return; }
    }
    const int slices = g.c / SLICE;
    if (mode == 2){
        static int sms = 0;
        if (!sms){
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaFuncSetAttribute(GroupedWgrad<T, CG>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        }
        WgTile tl = MakeWgTile(g);
        const int smem = (32 + tl.xr * tl.xc) * SLICE * (int)sizeof(float);
        // 每个 SM 约 4 个 block，分给各个切片
        const int blocks = std::max(1, std::min(tl.tiles, (sms * 4 + slices - 1) / slices));
        GroupedWgrad<T, CG><<<dim3(blocks, slices), THREADS, smem>>>(b, a, (float*)out, g, tl);
        return;
    }
    const int pixels = mode == 0 ? g.n * g.p * g.q : g.n * g.h * g.w;
    const int smem = SLICE * g.r * g.s * CG * (int)sizeof(float);
    const int chunks = (pixels + 32 * PIX_PER_THREAD - 1) / (32 * PIX_PER_THREAD);
    auto kern = mode == 0 ? GroupedDirect<T, CG, false> : GroupedDirect<T, CG, true>;
    // 常驻 block 数 = 每个 SM 能同时放下的 block 数 x SM 数，分给各个切片（每个 block 只读一次权重）
    int sms = 0, dev = 0, per_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, THREADS, smem);
    const int bx = std::max(1, std::min(chunks, (sms * std::max(1, per_sm) + slices - 1) / slices));
    kern<<<dim3(bx, slices), THREADS, smem>>>(a, b, (T*)out, g);
}

}  // namespace

bool ConvGroupedSupported(const ConvGeom& g){
    const int cg = g.c / g.groups;
    return g.groups > 1 && g.c == g.k && (cg == 4 || cg == 8 || cg == 16) && g.c % SLICE == 0 && g.r * g.s <= 12;
}

// mode 0：a = x, b = w, out = y；mode 1：a = dy, b = w, out = dx；mode 2：a = dy, b = x, out = dw（fp32，累加）
template <typename T>
void ConvGrouped(int mode, const T* a, const T* b, void* out, const ConvGeom& g){
    const int cg = g.c / g.groups;
    if (cg == 4) Run<T, 4>(mode, a, b, out, g);
    else if (cg == 8) Run<T, 8>(mode, a, b, out, g);
    else Run<T, 16>(mode, a, b, out, g);
}

template void ConvGrouped<float>(int, const float*, const float*, void*, const ConvGeom&);
template void ConvGrouped<bf16>(int, const bf16*, const bf16*, void*, const ConvGeom&);
