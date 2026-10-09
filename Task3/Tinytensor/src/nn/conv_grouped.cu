// 每组通道很少的分组卷积（ResNeXt 的 3x3，每组 4 / 8 / 16 个通道）：直接卷积，不走 GEMM
// 这类卷积的计算量很小（每个输出只有 R*S*CG 次乘加），瓶颈在访存；用 GEMM 做会浪费大部分 tile。
// 要求：输入输出通道数相同（C == K）、每组输入输出通道数相同（cg == kg = CG）、C 是 64 的倍数
// 一个 block 负责 64 个通道（及其权重，整块放进共享内存），block 常驻、依次处理多段像素：
//   前向 / 输入梯度：warp w 负责其中 8 个通道，lane 对应像素；权重在 warp 内是统一地址（共享内存广播）
//   权重梯度：dy 与输入窗口先读进共享内存，线程 = (通道, 抽头组) 沿像素累加，最后 atomicAdd
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

template <typename T, int CG>
void Run(int mode, const T* a, const T* b, void* out, const ConvGeom& g){
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
