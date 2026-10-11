// 池化、逐元素运算、ViT token 处理、损失、GPU 数据增强、优化器
#include <cmath>
#include <initializer_list>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

// ---------- 池化（NHWC）----------
template <typename T>
__global__ void MaxPool2x2Fwd(const T* __restrict__ x, T* __restrict__ y, unsigned char* __restrict__ arg,
                              int N, int H, int W, int C){
    const int P = H / 2, Q = W / 2;
    const size_t total = (size_t)N * P * Q * C;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % C; size_t t = i / C; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
        const T* base = x + (((size_t)n * H + 2 * p) * W + 2 * q) * C + c;
        float best = ToF(base[0]); int which = 0;
        float v1 = ToF(base[C]), v2 = ToF(base[(size_t)W * C]), v3 = ToF(base[(size_t)W * C + C]);
        if (v1 > best){ best = v1; which = 1; }
        if (v2 > best){ best = v2; which = 2; }
        if (v3 > best){ best = v3; which = 3; }
        y[i] = FromF<T>(best);
        arg[i] = (unsigned char)which;
    }
}

// 每个输入元素只属于一个窗口：直接按 argmax 判断取不取梯度（不需要清零和 atomic）
template <typename T>
__global__ void MaxPool2x2Bwd(const T* __restrict__ dy, const unsigned char* __restrict__ arg, T* __restrict__ dx,
                              int N, int H, int W, int C){
    const int P = H / 2, Q = W / 2;
    const size_t total = (size_t)N * H * W * C;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % C; size_t t = i / C; int w = t % W; t /= W; int h = t % H; int n = t / H;
        int p = h / 2, q = w / 2;
        float g = 0.f;
        if (p < P && q < Q){
            size_t j = (((size_t)n * P + p) * Q + q) * C + c;
            if (arg[j] == ((h & 1) * 2 + (w & 1))) g = ToF(dy[j]);
        }
        dx[i] = FromF<T>(g);
    }
}

// ---------- 向量化版本：每个线程处理 8 个连续通道（bf16 一次 16 字节）。条件：C % 8 == 0、H / W 为偶数、指针 16 字节对齐 ----------
constexpr int VEC = 8;
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
bool Aligned16(std::initializer_list<const void*> ptrs){
    for (const void* p : ptrs) if ((size_t)p % 16 != 0) return false;
    return true;
}

// 每个线程负责一个 2x2 窗口的 8 个通道
template <typename T>
__global__ void MaxPool2x2FwdVec(const T* __restrict__ x, T* __restrict__ y, unsigned char* __restrict__ arg,
                                 int N, int H, int W, int C){
    const int P = H / 2, Q = W / 2, CV = C / VEC;
    const size_t total = (size_t)N * P * Q * CV;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int cv = i % CV; size_t t = i / CV; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
        const T* base = x + (((size_t)n * H + 2 * p) * W + 2 * q) * C + cv * VEC;
        float v[4][VEC];
        Load8(base, v[0]); Load8(base + C, v[1]); Load8(base + (size_t)W * C, v[2]); Load8(base + (size_t)W * C + C, v[3]);
        float best[VEC];
        unsigned char which[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j){
            best[j] = v[0][j]; which[j] = 0;
            #pragma unroll
            for (int k = 1; k < 4; ++k) if (v[k][j] > best[j]){ best[j] = v[k][j]; which[j] = k; }
        }
        const size_t o = i * VEC;
        Store8(y + o, best);
        *(uint2*)(arg + o) = *(const uint2*)which;
    }
}

// 每个线程负责一个窗口的 8 个通道：读一次 dy / argmax，写 4 个位置
template <typename T>
__global__ void MaxPool2x2BwdVec(const T* __restrict__ dy, const unsigned char* __restrict__ arg, T* __restrict__ dx,
                                 int N, int H, int W, int C){
    const int P = H / 2, Q = W / 2, CV = C / VEC;
    const size_t total = (size_t)N * P * Q * CV;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int cv = i % CV; size_t t = i / CV; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
        float g[VEC];
        Load8(dy + i * VEC, g);
        uint2 a2 = __ldg((const uint2*)(arg + i * VEC));
        const unsigned char* a = (const unsigned char*)&a2;
        T* base = dx + (((size_t)n * H + 2 * p) * W + 2 * q) * C + cv * VEC;
        #pragma unroll
        for (int k = 0; k < 4; ++k){
            float o[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) o[j] = a[j] == k ? g[j] : 0.f;
            Store8(base + (k >> 1) * (size_t)W * C + (k & 1) * C, o);
        }
    }
}

template <typename T>
__global__ void AddKernelVec(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, size_t nv){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nv; i += (size_t)gridDim.x * blockDim.x){
        float x[VEC], z[VEC];
        Load8(a + i * VEC, x); Load8(b + i * VEC, z);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) x[j] += z[j];
        Store8(out + i * VEC, x);
    }
}

// [N, HW, C] -> [N, C]：每个线程负责一个 (n, c)，沿 HW 累加（相邻线程读相邻通道，合并访存）
template <typename T>
__global__ void GapFwd(const T* __restrict__ x, T* __restrict__ y, int N, int HW, int C){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    const int n = i / C, c = i % C;
    const T* p = x + (size_t)n * HW * C + c;
    float s = 0.f;
    for (int k = 0; k < HW; ++k) s += ToF(p[(size_t)k * C]);
    y[i] = FromF<T>(s / HW);
}

template <typename T>
__global__ void GapBwd(const T* __restrict__ dy, T* __restrict__ dx, int N, int HW, int C){
    const size_t total = (size_t)N * HW * C;
    const float inv = 1.f / HW;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % C; int n = i / ((size_t)HW * C);
        dx[i] = FromF<T>(ToF(dy[(size_t)n * C + c]) * inv);
    }
}

// ---------- 逐元素 ----------
template <typename T>
__global__ void AddKernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, size_t n){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = FromF<T>(ToF(a[i]) + ToF(b[i]));
}

template <typename Src, typename Dst>
__global__ void CastKernel(const Src* __restrict__ a, Dst* __restrict__ b, size_t n){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b[i] = FromF<Dst>(ToF(a[i]));
}

// ---------- ViT token ----------
template <typename T>
__global__ void TokensFwd(const T* __restrict__ patches, const T* __restrict__ cls, const T* __restrict__ pos,
                          T* __restrict__ out, int B, int Np, int D){
    const size_t total = (size_t)B * (Np + 1) * D;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int d = i % D; size_t t = i / D; int l = t % (Np + 1); int b = t / (Np + 1);
        float v = l == 0 ? ToF(cls[d]) : ToF(patches[((size_t)b * Np + l - 1) * D + d]);
        out[i] = FromF<T>(v + ToF(pos[(size_t)l * D + d]));
    }
}

// dpatches = dout[:, 1:]；dpos[l, d] += sum_b dout[b, l, d]；dcls[d] += sum_b dout[b, 0, d]
template <typename T>
__global__ void TokensBwd(const T* __restrict__ dout, T* __restrict__ dpatches, float* __restrict__ dcls,
                          float* __restrict__ dpos, int B, int Np, int D){
    const int L = Np + 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // 一个线程负责一个 (l, d)
    if (i >= L * D) return;
    const int l = i / D, d = i % D;
    float s = 0.f;
    for (int b = 0; b < B; ++b){
        float g = ToF(dout[((size_t)b * L + l) * D + d]);
        s += g;
        if (l > 0) dpatches[((size_t)b * Np + l - 1) * D + d] = FromF<T>(g);
    }
    dpos[i] += s;
    if (l == 0) dcls[d] += s;
}

template <typename T>
__global__ void SelectFwd(const T* __restrict__ x, T* __restrict__ y, int B, int L, int D){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * D) y[i] = x[(size_t)(i / D) * L * D + i % D];
}

template <typename T>
__global__ void SelectBwd(const T* __restrict__ dy, T* __restrict__ dx, int B, int L, int D){
    const size_t total = (size_t)B * L * D;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int d = i % D; size_t t = i / D; int l = t % L; int b = t / L;
        dx[i] = l == 0 ? dy[(size_t)b * D + d] : FromF<T>(0.f);
    }
}

// ---------- softmax 交叉熵：一个 warp 处理一行 ----------
// 软标签（训练时的正则化，都可以关）：target = (1 - eps) * (lam * onehot(y) + (1 - lam) * onehot(y2)) + eps / C
//   eps：label smoothing；y2 / lam：Mixup / CutMix 的另一个样本的标签与混合比例（lam 是 GPU 上的标量，由 load_batch 写）
// loss = logsumexp(z) - sum_i target_i * z_i，grad = softmax - target；准确率按主标签 y 统计
__global__ void SoftmaxCE(const float* __restrict__ logits, const int* __restrict__ labels, float* __restrict__ grad,
                          float grad_scale, float* __restrict__ loss_sum, int* __restrict__ correct,
                          int N, int C, bool want_grad, const int* __restrict__ labels2 = nullptr,
                          const float* __restrict__ lam_ptr = nullptr, float eps = 0.f){
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const float* z = logits + (size_t)row * C;
    float mx = -INFINITY; int arg = 0;
    for (int i = lane; i < C; i += 32){ float v = z[i]; if (v > mx){ mx = v; arg = i; } }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1){
        float om = __shfl_xor_sync(0xffffffff, mx, o);
        int oa = __shfl_xor_sync(0xffffffff, arg, o);
        if (om > mx || (om == mx && oa < arg)){ mx = om; arg = oa; }
    }
    float s = 0.f;
    for (int i = lane; i < C; i += 32) s += __expf(z[i] - mx);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    const int y = labels[row];
    const float lam = lam_ptr ? *lam_ptr : 1.f;
    const int y2 = labels2 ? labels2[row] : y;
    const float w1 = (1.f - eps) * lam, w2 = (1.f - eps) * (1.f - lam), wu = eps / C;
    if (want_grad){
        const float inv = 1.f / s;
        for (int i = lane; i < C; i += 32){
            const float t = (i == y ? w1 : 0.f) + (i == y2 ? w2 : 0.f) + wu;
            grad[(size_t)row * C + i] = (__expf(z[i] - mx) * inv - t) * grad_scale;
        }
    }
    float zsum = 0.f;
    if (eps != 0.f){
        for (int i = lane; i < C; i += 32) zsum += z[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) zsum += __shfl_xor_sync(0xffffffff, zsum, o);
    }
    if (lane == 0){
        const float lse = logf(s) + mx;
        atomicAdd(loss_sum, lse - w1 * z[y] - w2 * z[y2] - wu * zsum);
        if (arg == y) atomicAdd(correct, 1);
    }
}

// ---------- 数据增强 ----------
struct Norm3{ float mean[3]; float inv_std[3]; };

// Mixup / CutMix（每个 step 一组参数，由 hash(seed, 全局步数) 决定；PyTorch 对照实现用完全相同的整数运算）：
//   以 p16 / 65536 的概率混合；两者都开时各一半。第 b 个样本与第 B-1-b 个样本（各自做自己的裁剪翻转）混合
//   Mixup：x = lam * x_b + (1 - lam) * x_{B-1-b}，lam ~ U(0,1)
//   CutMix：边长 s = isqrt(floor(u * H * W)) 的方框（中心均匀，超出图像的部分截掉）内取另一个样本，lam = 1 - 框面积 / (H*W)
struct MixCfg{ int p16; bool mixup, cutmix; };
struct MixStep{ int mode, y0, y1, x0, x1; float lam; };     // mode 0 不混合，1 Mixup，2 CutMix

// lowbias32（Chris Wellons），PyTorch 对照实现里用完全相同的整数运算
__host__ __device__ __forceinline__ unsigned int HashHD(unsigned int x){
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ MixStep MixParams(const MixCfg& m, unsigned int seed, unsigned int t, int H, int W){
    MixStep r{0, 0, 0, 0, 0, 1.f};
    if (!m.mixup && !m.cutmix) return r;
    const unsigned int h0 = HashHD(seed ^ HashHD(t * 0x85EBCA6BU + 0x632BE5ABU));
    if ((int)(h0 & 0xFFFFu) >= m.p16) return r;
    const bool cut = m.mixup && m.cutmix ? ((h0 >> 16) & 1u) : m.cutmix;
    const unsigned int h1 = HashHD(h0 ^ 0x27D4EB2FU), h2 = HashHD(h1 ^ 0x165667B1U);
    const unsigned int u24 = h1 >> 8;
    if (!cut){
        r.mode = 1;
        r.lam = (float)u24 * (1.f / 16777216.f);
        return r;
    }
    const long long target = ((long long)u24 * (H * W)) >> 24;
    long long s = (long long)sqrt((double)target);
    while (s * s > target) --s;
    while ((s + 1) * (s + 1) <= target) ++s;
    const int cy = (int)(h2 % (unsigned int)H), cx = (int)((h2 >> 16) % (unsigned int)W);
    r.mode = 2;
    r.y0 = max(cy - (int)s / 2, 0); r.y1 = min(cy - (int)s / 2 + (int)s, H);
    r.x0 = max(cx - (int)s / 2, 0); r.x1 = min(cx - (int)s / 2 + (int)s, W);
    r.lam = 1.f - __fdiv_rn((float)((r.y1 - r.y0) * (r.x1 - r.x0)), (float)(H * W));
    return r;
}

// 第 b 个样本经过裁剪翻转后 (h, w, c) 处的值（归一化前，[0,1]）
__device__ __forceinline__ float AugPixel(const unsigned char* __restrict__ images, int idx, int b, int h, int w, int c,
                                          int H, int W, bool augment, int pad, unsigned int seed, unsigned int t){
    int sh = h, sw = w;
    if (augment){
        unsigned int rnd = HashHD(seed ^ HashHD(t * 0x9E3779B9U + (unsigned int)b));
        int range = 2 * pad + 1;
        int dy = (int)(rnd % range) - pad;
        int dx = (int)((rnd >> 8) % range) - pad;
        bool flip = (rnd >> 16) & 1;
        sh = h + dy;
        sw = (flip ? W - 1 - w : w) + dx;
    }
    if (sh >= 0 && sh < H && sw >= 0 && sw < W) return images[(((size_t)idx * H + sh) * W + sw) * 3 + c] * (1.f / 255.f);
    return 0.f;
}

// 一个线程写一个输出元素；最后一个完成的 block 推进步数（与 MNIST 的 gather_xy 相同的做法）
template <typename T>
__global__ void LoadBatchKernel(const unsigned char* __restrict__ images, const int* __restrict__ labels,
                                const int* __restrict__ order, int* __restrict__ step, T* __restrict__ x,
                                int* __restrict__ y, int batch, int H, int W, Norm3 norm, bool augment, int pad,
                                unsigned int seed, MixCfg mix = MixCfg{0, false, false}, int* __restrict__ y2 = nullptr,
                                float* __restrict__ lam_out = nullptr){
    const int base = step[0] * batch;
    const unsigned int t = (unsigned int)step[2];
    const MixStep ms = MixParams(mix, seed, t, H, W);
    const size_t total = (size_t)batch * H * W * 3;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        int c = i % 3; size_t r = i / 3; int w = r % W; r /= W; int h = r % H; int b = r / H;
        const int b2 = batch - 1 - b;
        float v;
        if (ms.mode == 2 && h >= ms.y0 && h < ms.y1 && w >= ms.x0 && w < ms.x1)
            v = AugPixel(images, order[base + b2], b2, h, w, c, H, W, augment, pad, seed, t);
        else
            v = AugPixel(images, order[base + b], b, h, w, c, H, W, augment, pad, seed, t);
        float xn = __fmul_rn(v - norm.mean[c], norm.inv_std[c]);
        if (ms.mode == 1){
            const float v2 = AugPixel(images, order[base + b2], b2, h, w, c, H, W, augment, pad, seed, t);
            const float x2 = __fmul_rn(v2 - norm.mean[c], norm.inv_std[c]);
            xn = __fadd_rn(__fmul_rn(ms.lam, xn), __fmul_rn(1.f - ms.lam, x2));
        }
        x[i] = FromF<T>(xn);
    }
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch; b += gridDim.x * blockDim.x){
        y[b] = labels[order[base + b]];
        if (y2) y2[b] = labels[order[base + batch - 1 - b]];
    }
    if (lam_out && blockIdx.x == 0 && threadIdx.x == 0) *lam_out = ms.lam;
    __syncthreads();
    if (threadIdx.x == 0){
        __threadfence();
        if (atomicAdd(step + 1, 1) == gridDim.x - 1){
            step[0] += 1;
            step[2] += 1;
            step[1] = 0;
        }
    }
}

// ---------- 优化器 ----------
__device__ __forceinline__ float LrAt(const LrSchedule& s, int t){
    if (t < s.warmup_steps) return s.base_lr * (t + 1) / s.warmup_steps;
    float progress = s.total_steps > s.warmup_steps
        ? fminf(1.f, (float)(t - s.warmup_steps) / (s.total_steps - s.warmup_steps)) : 1.f;
    float cosine = 0.5f * (1.f + cospif(progress));
    return s.base_lr * (s.final_ratio + (1.f - s.final_ratio) * cosine);
}

__global__ void SgdKernel(float* __restrict__ w, const float* __restrict__ g, float* __restrict__ m, bf16* __restrict__ wl,
                          size_t n, const int* __restrict__ step, LrSchedule s, float mu, float wd, bool nesterov){
    const float lr = LrAt(s, step[2] - 1);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x){
        float wi = w[i];
        float gi = g[i] + wd * wi;
        float mi = mu * m[i] + gi;
        m[i] = mi;
        wi -= lr * (nesterov ? gi + mu * mi : mi);
        w[i] = wi;
        if (wl) wl[i] = __float2bfloat16(wi);
    }
}

__global__ void AdamWKernel(float* __restrict__ w, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v,
                            bf16* __restrict__ wl, size_t n, const int* __restrict__ step, LrSchedule s,
                            float b1, float b2, float eps, float wd){
    const int t = step[2];   // 本步是第 t 步（从 1 开始）
    const float lr = LrAt(s, t - 1);
    const float c1 = 1.f - powf(b1, (float)t), c2 = 1.f - powf(b2, (float)t);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x){
        float wi = w[i] * (1.f - lr * wd);
        float gi = g[i];
        float mi = b1 * m[i] + (1.f - b1) * gi;
        float vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        wi -= lr * (mi / c1) / (sqrtf(vi / c2) + eps);
        w[i] = wi;
        if (wl) wl[i] = __float2bfloat16(wi);
    }
}

int Capped(size_t n){ return std::min(GridFor(n, 256), 24 * 32); }

// 单独的激活（不融合时用）：act 1 = relu，2 = gelu（erf 形式，与 PyTorch 默认一致）
__device__ __forceinline__ float ActF(float z, int act){
    return act == 1 ? fmaxf(z, 0.f) : 0.5f * z * (1.f + erff(z * 0.70710678f));
}
__device__ __forceinline__ float ActGrad(float z, int act){
    if (act == 1) return z > 0.f ? 1.f : 0.f;
    const float cdf = 0.5f * (1.f + erff(z * 0.70710678f));
    return cdf + z * 0.39894228f * __expf(-0.5f * z * z);
}
template <typename T>
__global__ void ActFwdKernel(const T* __restrict__ x, T* __restrict__ y, size_t n, int act){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        y[i] = FromF<T>(ActF(ToF(x[i]), act));
}
template <typename T>
__global__ void ActBwdKernel(const T* __restrict__ x, const T* __restrict__ dy, T* __restrict__ dx, size_t n, int act){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dx[i] = FromF<T>(ToF(dy[i]) * ActGrad(ToF(x[i]), act));
}

}  // namespace

template <typename T>
void MaxPool2x2Forward(const TinyTensor<T>& x, TinyTensor<T>& y, TinyTensor<unsigned char>& argmax){
    NNCheckGpu(x, "maxpool"); NNCheck(x.shape.size() == 4, "maxpool: NHWC");
    const int N = x.shape[0], H = x.shape[1], W = x.shape[2], C = x.shape[3];
    y.resize_discard({N, H / 2, W / 2, C});
    argmax.resize_discard({N, H / 2, W / 2, C});
    if (C % VEC == 0 && H % 2 == 0 && W % 2 == 0 && Aligned16({x.p_data, y.p_data}) && (size_t)argmax.p_data % 8 == 0){
        MaxPool2x2FwdVec<T><<<Capped(y.size / VEC), 256>>>(x.p_data, y.p_data, argmax.p_data, N, H, W, C);
        return;
    }
    MaxPool2x2Fwd<T><<<Capped(y.size), 256>>>(x.p_data, y.p_data, argmax.p_data, N, H, W, C);
}

template <typename T>
void MaxPool2x2Backward(const TinyTensor<T>& dy, const TinyTensor<unsigned char>& argmax, TinyTensor<T>& dx,
                        const std::vector<int>& x_shape){
    NNCheckGpu(dy, "maxpool_backward");
    const int N = x_shape[0], H = x_shape[1], W = x_shape[2], C = x_shape[3];
    dx.resize_discard(x_shape);
    if (C % VEC == 0 && H % 2 == 0 && W % 2 == 0 && Aligned16({dy.p_data, dx.p_data}) && (size_t)argmax.p_data % 8 == 0){
        MaxPool2x2BwdVec<T><<<Capped(dy.size / VEC), 256>>>(dy.p_data, argmax.p_data, dx.p_data, N, H, W, C);
        return;
    }
    MaxPool2x2Bwd<T><<<Capped(dx.size), 256>>>(dy.p_data, argmax.p_data, dx.p_data, N, H, W, C);
}

template <typename T>
void GlobalAvgPoolForward(const TinyTensor<T>& x, TinyTensor<T>& y){
    NNCheckGpu(x, "avgpool"); NNCheck(x.shape.size() == 4, "avgpool: NHWC");
    const int N = x.shape[0], HW = x.shape[1] * x.shape[2], C = x.shape[3];
    y.resize_discard({N, C});
    GapFwd<T><<<(N * C + 255) / 256, 256>>>(x.p_data, y.p_data, N, HW, C);
}

template <typename T>
void GlobalAvgPoolBackward(const TinyTensor<T>& dy, TinyTensor<T>& dx, const std::vector<int>& x_shape){
    NNCheckGpu(dy, "avgpool_backward");
    const int N = x_shape[0], HW = x_shape[1] * x_shape[2], C = x_shape[3];
    dx.resize_discard(x_shape);
    GapBwd<T><<<Capped(dx.size), 256>>>(dy.p_data, dx.p_data, N, HW, C);
}

template <typename T>
void AddForward(const TinyTensor<T>& a, const TinyTensor<T>& b, TinyTensor<T>& out){
    NNCheckGpu(a, "add"); NNCheck(a.size == b.size, "add: size mismatch");
    out.resize_discard(a.shape);
    if (a.size % VEC == 0 && Aligned16({a.p_data, b.p_data, out.p_data})){
        AddKernelVec<T><<<Capped(a.size / VEC), 256>>>(a.p_data, b.p_data, out.p_data, a.size / VEC);
        return;
    }
    AddKernel<T><<<Capped(a.size), 256>>>(a.p_data, b.p_data, out.p_data, a.size);
}

template <typename T>
void ActForward(const TinyTensor<T>& x, TinyTensor<T>& y, int act){
    NNCheckGpu(x, "act"); NNCheck(act == 1 || act == 2, "act: 1 relu / 2 gelu");
    y.resize_discard(x.shape);
    ActFwdKernel<T><<<Capped(x.size), 256>>>(x.p_data, y.p_data, x.size, act);
}

template <typename T>
void ActBackward(const TinyTensor<T>& x, const TinyTensor<T>& dy, TinyTensor<T>& dx, int act){
    NNCheckGpu(x, "act_backward"); NNCheck(dy.size == x.size, "act_backward: size mismatch");
    dx.resize_discard(x.shape);
    ActBwdKernel<T><<<Capped(x.size), 256>>>(x.p_data, dy.p_data, dx.p_data, x.size, act);
}

template <typename Src, typename Dst>
void CastTensor(const TinyTensor<Src>& src, TinyTensor<Dst>& dst){
    NNCheckGpu(src, "cast");
    dst.resize_discard(src.shape);
    CastKernel<Src, Dst><<<Capped(src.size), 256>>>(src.p_data, dst.p_data, src.size);
}

template <typename T>
void TokensForward(const TinyTensor<T>& patches, const TinyTensor<T>& cls, const TinyTensor<T>& pos, TinyTensor<T>& out){
    NNCheck(patches.shape.size() == 3, "tokens: patches [B, N, D]");
    const int B = patches.shape[0], Np = patches.shape[1], D = patches.shape[2];
    NNCheck(cls.size == (size_t)D && pos.size == (size_t)(Np + 1) * D, "tokens: cls / pos size");
    out.resize_discard({B, Np + 1, D});
    TokensFwd<T><<<Capped(out.size), 256>>>(patches.p_data, cls.p_data, pos.p_data, out.p_data, B, Np, D);
}

template <typename T>
void TokensBackward(const TinyTensor<T>& dout, TinyTensor<T>& dpatches, TinyTensor<float>& dcls, TinyTensor<float>& dpos){
    const int B = dout.shape[0], L = dout.shape[1], D = dout.shape[2];
    dpatches.resize_discard({B, L - 1, D});
    TokensBwd<T><<<(L * D + 255) / 256, 256>>>(dout.p_data, dpatches.p_data, dcls.p_data, dpos.p_data, B, L - 1, D);
}

template <typename T>
void SelectTokenForward(const TinyTensor<T>& x, TinyTensor<T>& y){
    const int B = x.shape[0], L = x.shape[1], D = x.shape[2];
    y.resize_discard({B, D});
    SelectFwd<T><<<(B * D + 255) / 256, 256>>>(x.p_data, y.p_data, B, L, D);
}

template <typename T>
void SelectTokenBackward(const TinyTensor<T>& dy, TinyTensor<T>& dx, const std::vector<int>& x_shape){
    const int B = x_shape[0], L = x_shape[1], D = x_shape[2];
    dx.resize_discard(x_shape);
    SelectBwd<T><<<Capped(dx.size), 256>>>(dy.p_data, dx.p_data, B, L, D);
}

void SoftmaxCrossEntropyForwardBackward(const TinyTensor<float>& logits, const TinyTensor<int>& labels,
                                        TinyTensor<float>& grad, float grad_scale,
                                        TinyTensor<float>& loss_sum, TinyTensor<int>& correct, bool want_grad){
    NNCheck(logits.shape.size() == 2, "softmax_ce: logits [N, C]");
    const int N = logits.shape[0], C = logits.shape[1];
    NNCheck(labels.size == (size_t)N, "softmax_ce: labels size");
    if (want_grad) grad.resize_discard(logits.shape);
    SoftmaxCE<<<(N * 32 + 255) / 256, 256>>>(logits.p_data, labels.p_data, want_grad ? grad.p_data : nullptr, grad_scale,
                                             loss_sum.p_data, correct.p_data, N, C, want_grad);
}

void SoftmaxCrossEntropyMix(const TinyTensor<float>& logits, const TinyTensor<int>& labels,
                            const TinyTensor<int>& labels2, const TinyTensor<float>& lam, float eps,
                            TinyTensor<float>& grad, float grad_scale,
                            TinyTensor<float>& loss_sum, TinyTensor<int>& correct){
    NNCheck(logits.shape.size() == 2, "softmax_ce_mix: logits [N, C]");
    const int N = logits.shape[0], C = logits.shape[1];
    NNCheck(labels.size == (size_t)N && labels2.size == (size_t)N && lam.size >= 1, "softmax_ce_mix: labels / lam");
    grad.resize_discard(logits.shape);
    SoftmaxCE<<<(N * 32 + 255) / 256, 256>>>(logits.p_data, labels.p_data, grad.p_data, grad_scale,
                                             loss_sum.p_data, correct.p_data, N, C, true, labels2.p_data,
                                             lam.p_data, eps);
}

template <typename T>
void LoadBatch(const TinyTensor<unsigned char>& images, const TinyTensor<int>& labels, const TinyTensor<int>& order,
               TinyTensor<int>& step, TinyTensor<T>& x, TinyTensor<int>& y, int batch,
               const std::vector<float>& mean, const std::vector<float>& std, bool augment, int pad,
               unsigned int seed){
    NNCheck(images.shape.size() == 4 && images.shape[3] == 3, "load_batch: images [N, H, W, 3] uint8");
    NNCheck(step.size >= 4, "load_batch: step needs 4 ints");
    NNCheck(mean.size() == 3 && std.size() == 3, "load_batch: mean/std");
    const int H = images.shape[1], W = images.shape[2];
    NNCheck(x.shape == std::vector<int>({batch, H, W, 3}) && y.size == (size_t)batch, "load_batch: output shape");
    Norm3 norm;
    for (int c = 0; c < 3; ++c){ norm.mean[c] = mean[c]; norm.inv_std[c] = 1.f / std[c]; }
    LoadBatchKernel<T><<<Capped(x.size), 256>>>(images.p_data, labels.p_data, order.p_data, step.p_data, x.p_data,
                                                y.p_data, batch, H, W, norm, augment, pad, seed);
}

template <typename T>
void LoadBatchMix(const TinyTensor<unsigned char>& images, const TinyTensor<int>& labels, const TinyTensor<int>& order,
                  TinyTensor<int>& step, TinyTensor<T>& x, TinyTensor<int>& y, TinyTensor<int>& y2,
                  TinyTensor<float>& lam, int batch, const std::vector<float>& mean, const std::vector<float>& std,
                  int pad, unsigned int seed, float mix_prob, bool mixup, bool cutmix){
    NNCheck(images.shape.size() == 4 && images.shape[3] == 3, "load_batch_mix: images [N, H, W, 3] uint8");
    NNCheck(step.size >= 4, "load_batch_mix: step needs 4 ints");
    NNCheck(mean.size() == 3 && std.size() == 3, "load_batch_mix: mean/std");
    const int H = images.shape[1], W = images.shape[2];
    NNCheck(x.shape == std::vector<int>({batch, H, W, 3}) && y.size == (size_t)batch && y2.size == (size_t)batch &&
            lam.size >= 1, "load_batch_mix: output shape");
    Norm3 norm;
    for (int c = 0; c < 3; ++c){ norm.mean[c] = mean[c]; norm.inv_std[c] = 1.f / std[c]; }
    MixCfg mix{(int)lroundf(mix_prob * 65536.f), mixup, cutmix};
    LoadBatchKernel<T><<<Capped(x.size), 256>>>(images.p_data, labels.p_data, order.p_data, step.p_data, x.p_data,
                                                y.p_data, batch, H, W, norm, true, pad, seed, mix, y2.p_data,
                                                lam.p_data);
}

void SgdMomentumStep(TinyTensor<float>& w, const TinyTensor<float>& g, TinyTensor<float>& m, TinyTensor<bf16>* w_lowp,
                     const TinyTensor<int>& step, LrSchedule lr, float momentum, float weight_decay, bool nesterov){
    NNCheck(w.size == g.size && w.size == m.size && (!w_lowp || w_lowp->size == w.size), "sgd: size mismatch");
    SgdKernel<<<Capped(w.size), 256>>>(w.p_data, g.p_data, m.p_data, w_lowp ? w_lowp->p_data : nullptr, w.size,
                                       step.p_data, lr, momentum, weight_decay, nesterov);
}

void AdamWStep(TinyTensor<float>& w, const TinyTensor<float>& g, TinyTensor<float>& m, TinyTensor<float>& v,
               TinyTensor<bf16>* w_lowp, const TinyTensor<int>& step, LrSchedule lr, float beta1, float beta2,
               float eps, float weight_decay){
    NNCheck(w.size == g.size && w.size == m.size && w.size == v.size, "adamw: size mismatch");
    AdamWKernel<<<Capped(w.size), 256>>>(w.p_data, g.p_data, m.p_data, v.p_data, w_lowp ? w_lowp->p_data : nullptr,
                                         w.size, step.p_data, lr, beta1, beta2, eps, weight_decay);
}

#define INSTANTIATE(T) \
    template void MaxPool2x2Forward<T>(const TinyTensor<T>&, TinyTensor<T>&, TinyTensor<unsigned char>&); \
    template void MaxPool2x2Backward<T>(const TinyTensor<T>&, const TinyTensor<unsigned char>&, TinyTensor<T>&, \
                                        const std::vector<int>&); \
    template void GlobalAvgPoolForward<T>(const TinyTensor<T>&, TinyTensor<T>&); \
    template void GlobalAvgPoolBackward<T>(const TinyTensor<T>&, TinyTensor<T>&, const std::vector<int>&); \
    template void AddForward<T>(const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<T>&);     template void ActForward<T>(const TinyTensor<T>&, TinyTensor<T>&, int);     template void ActBackward<T>(const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<T>&, int); \
    template void TokensForward<T>(const TinyTensor<T>&, const TinyTensor<T>&, const TinyTensor<T>&, TinyTensor<T>&); \
    template void TokensBackward<T>(const TinyTensor<T>&, TinyTensor<T>&, TinyTensor<float>&, TinyTensor<float>&); \
    template void SelectTokenForward<T>(const TinyTensor<T>&, TinyTensor<T>&); \
    template void SelectTokenBackward<T>(const TinyTensor<T>&, TinyTensor<T>&, const std::vector<int>&); \
    template void LoadBatch<T>(const TinyTensor<unsigned char>&, const TinyTensor<int>&, const TinyTensor<int>&, \
                               TinyTensor<int>&, TinyTensor<T>&, TinyTensor<int>&, int, const std::vector<float>&, \
                               const std::vector<float>&, bool, int, unsigned int); \
    template void LoadBatchMix<T>(const TinyTensor<unsigned char>&, const TinyTensor<int>&, const TinyTensor<int>&, \
                                  TinyTensor<int>&, TinyTensor<T>&, TinyTensor<int>&, TinyTensor<int>&, TinyTensor<float>&, \
                                  int, const std::vector<float>&, const std::vector<float>&, int, unsigned int, float, \
                                  bool, bool);
INSTANTIATE(float)
INSTANTIATE(bf16)
template void CastTensor<float, bf16>(const TinyTensor<float>&, TinyTensor<bf16>&);
template void CastTensor<bf16, float>(const TinyTensor<bf16>&, TinyTensor<float>&);
template void CastTensor<float, float>(const TinyTensor<float>&, TinyTensor<float>&);
