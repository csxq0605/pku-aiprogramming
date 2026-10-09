#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include "base.h"
#include "Layers_kernels.cuh"

// cuBLAS 句柄创建代价很高，全局只创建一次
// 绑定到 per-thread stream，使 cuBLAS 调用可以被 CUDA Graph 录制
// workspace 设为 0：禁止 split-K。split-K 每个 GEMM 会多一个 splitKreduce kernel，
// 而在 WSL 下每个 kernel 的提交延迟（约 4µs）比 split-K 省下的 GPU 时间更贵。
// 实测 CNN 每 step 28 -> 24 个 kernel，墙钟 0.219 -> 0.179 ms/step。可用 TT_CUBLAS_WS=字节数 覆盖。
static cublasHandle_t CublasHandle(){
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr){
        cublasCreate(&handle);
        cublasSetStream(handle, cudaStreamPerThread);
        static void* workspace = nullptr;
        const char* env = getenv("TT_CUBLAS_WS");
        const size_t workspace_bytes = env ? (size_t)atoll(env) : 0;
        if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);
        cublasSetWorkspace(handle, workspace, workspace_bytes);
    }
    return handle;
}

// 行主序矩阵等价于列主序的转置，因此计算 C^T = op(B)^T * op(A)^T
void cudaGemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    int lda = (transa == CUBLAS_OP_N) ? k : m;
    int ldb = (transb == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    cublasSgemm(CublasHandle(), transb, transa, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void cudaGemmStridedBatched(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float alpha,
    const float* A,
    long long stride_a,
    const float* B,
    long long stride_b,
    float beta,
    float* C,
    long long stride_c,
    int batch
){
    int lda = (transa == CUBLAS_OP_N) ? k : m;
    int ldb = (transb == CUBLAS_OP_N) ? n : k;
    int ldc = n;
    cublasSgemmStridedBatched(CublasHandle(), transb, transa, n, m, k, &alpha,
        B, ldb, stride_b, A, lda, stride_a, &beta, C, ldc, stride_c, batch);
}

// 一个线程负责 (样本, 输出位置, 输入通道) 的一个 kernel 窗口
// 每个样本的列矩阵形状为 [col_h * col_w, channels * kernel_h * kernel_w]
template <typename Type>
__global__ void cudaIm2Col(
    const Type* data_im,
    Type* data_col,
    const int kernels_num,
    const int channels,
    const int col_h,
    const int col_w,
    const int im_h,
    const int im_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    const int per_sample = channels * col_h * col_w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kernels_num; i += blockDim.x * gridDim.x){
        Type* data_col_ptr = data_col + (size_t)i * kernel_h * kernel_w;
        int index_batch = i / per_sample;
        int local = i % per_sample;
        int index_channel = local % channels;
        int index_kernel = local / channels;
        int index_row = index_kernel / col_w;
        int index_col = index_kernel % col_w;
        int im_row = index_row * stride_h - pad_h;
        int im_col = index_col * stride_w - pad_w;
        const Type* im = data_im + ((size_t)index_batch * channels + index_channel) * im_h * im_w;

        for (int row = 0; row < kernel_h; ++row){
            for (int col = 0; col < kernel_w; ++col){
                int row_im = im_row + row;
                int col_im = im_col + col;
                *data_col_ptr = (row_im >= 0 && row_im < im_h && col_im >= 0 && col_im < im_w) ? im[row_im * im_w + col_im] : Type(0);
                data_col_ptr++;
            }
        }
    }
}

// 一个线程负责输入图像中的一个像素，累加所有覆盖它的窗口
template <typename Type>
__global__ void cudaCol2Im(
    const Type* data_col,
    Type* data_im,
    const int kernels_num,
    const int channels,
    const int col_h,
    const int col_w,
    const int im_h,
    const int im_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    const int im_size = channels * im_h * im_w;
    const size_t col_size = (size_t)col_h * col_w * channels * kernel_h * kernel_w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kernels_num; i += blockDim.x * gridDim.x){
        int index_batch = i / im_size;
        int local = i % im_size;
        int index_w = local % im_w + pad_w;
        int index_h = (local / im_w) % im_h + pad_h;
        int index_channel = local / (im_w * im_h);
        int col_w_start = (index_w < kernel_w) ? 0 : (index_w - kernel_w) / stride_w + 1;
        int col_w_end = min(index_w / stride_w + 1, col_w);
        int col_h_start = (index_h < kernel_h) ? 0 : (index_h - kernel_h) / stride_h + 1;
        int col_h_end = min(index_h / stride_h + 1, col_h);
        const Type* col = data_col + index_batch * col_size;

        Type value = 0;
        for (int h_col = col_h_start; h_col < col_h_end; h_col++){
            for (int w_col = col_w_start; w_col < col_w_end; w_col++){
                int h_im = index_h - h_col * stride_h;
                int w_im = index_w - w_col * stride_w;
                int index_col = (((h_col * col_w + w_col) * channels + index_channel) * kernel_h + h_im) * kernel_w + w_im;
                value += col[index_col];
            }
        }
        data_im[i] = value;
    }
}

template <typename Type>
__global__ void cudaMaxPoolingForward(
    const Type* data_in,
    Type* data_out,
    Type* mask,
    const int kernels_num,
    const int channels,
    const int out_h,
    const int out_w,
    const int in_h,
    const int in_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kernels_num; i += blockDim.x * gridDim.x){
        int index_batch = i / out_h / out_w / channels;
        int index_channel = (i / out_h / out_w) % channels;
        int ph = (i / out_w) % out_h;
        int pw = i % out_w;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        int h_end = min(h_start + kernel_h, in_h);
        int w_end = min(w_start + kernel_w, in_w);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        Type max_val = -FLT_MAX;
        int max_idx = h_start * in_w + w_start;
        const Type* data_in_ptr = data_in + (index_batch * channels + index_channel) * in_h * in_w;
        for (int h = h_start; h < h_end; ++h){
            for (int w = w_start; w < w_end; ++w){
                if (data_in_ptr[h * in_w + w] > max_val){
                    max_val = data_in_ptr[h * in_w + w];
                    max_idx = h * in_w + w;
                }
            }
        }
        data_out[i] = max_val;
        mask[i] = max_idx;
    }
}

// 窗口可能重叠，用 atomicAdd 累加梯度
template <typename Type>
__global__ void cudaMaxPoolingBackward(
    const Type* grad_out,
    const Type* mask,
    Type* grad_in,
    const int kernels_num,
    const int channels,
    const int out_h,
    const int out_w,
    const int in_h,
    const int in_w
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kernels_num; i += blockDim.x * gridDim.x){
        int index_batch = i / out_h / out_w / channels;
        int index_channel = (i / out_h / out_w) % channels;
        int index_grad_in = (index_batch * channels + index_channel) * in_h * in_w + (int)mask[i];
        atomicAdd(grad_in + index_grad_in, grad_out[i]);
    }
}

template <typename Type>
__global__ void cudaAddBias(
    Type* data,
    const Type* bias,
    const int rows,
    const int cols,
    const int bias_size
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += blockDim.x * gridDim.x){
        data[i] += bias[bias_size == 1 ? 0 : i % cols];
    }
}

// 按列求和：block 为 32 列 x 8 行，每个线程累加同一列中间隔 8 行的元素，再在 block 内规约
template <typename Type>
__global__ void cudaColumnSum(
    const Type* data,
    Type* out,
    const int rows,
    const int cols
){
    __shared__ Type partial[8][33];
    const int j = blockIdx.x * 32 + threadIdx.x;
    Type sum = 0;
    if (j < cols){
        for (int i = threadIdx.y; i < rows; i += 8) sum += data[(size_t)i * cols + j];
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && j < cols){
        #pragma unroll
        for (int k = 1; k < 8; ++k) sum += partial[k][threadIdx.x];
        out[j] = sum;
    }
}

// 一个线程处理一行：减最大值、取指数、归一化
template <typename Type>
__global__ void cudaRowSoftmax(
    const Type* data_in,
    Type* data_out,
    const int rows,
    const int cols
){
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x){
        const Type* in = data_in + r * cols;
        Type* out = data_out + r * cols;
        Type max_val = in[0];
        for (int j = 1; j < cols; ++j){
            max_val = max(max_val, in[j]);
        }
        Type sum = 0;
        for (int j = 0; j < cols; ++j){
            out[j] = expf(in[j] - max_val);
            sum += out[j];
        }
        for (int j = 0; j < cols; ++j){
            out[j] /= sum;
        }
    }
}

template <typename Type>
__global__ void cudaChannelLog(
    const Type* data_in,
    Type* data_out,
    const int* labels,
    const int kernels_num,
    const int channels
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kernels_num; i += blockDim.x * gridDim.x){
        data_out[i] = -logf(fmaxf(data_in[i * channels + labels[i]], 1e-12f));
    }
}

template <typename Type>
__global__ void cudaSoftmaxGrad(
    const Type* softmax,
    const int* labels,
    Type* grad,
    const int rows,
    const int cols
){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += blockDim.x * gridDim.x){
        int r = i / cols;
        int c = i % cols;
        grad[i] = softmax[i] - (labels[r] == c ? Type(1) : Type(0));
    }
}

template <typename Type>
__global__ void cudaSumToScalar(const Type* data, Type* out, const int size, const Type scale){
    __shared__ Type partial[BLOCK_SIZE];
    Type sum = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x){
        sum += data[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        out[0] = partial[0] * scale;
    }
}

// FC 前向的 bias + ReLU：data = max(data + bias, 0)
template <typename Type>
__global__ void cudaAddBiasRelu(Type* data, const Type* bias, const int rows, const int cols){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += blockDim.x * gridDim.x){
        data[i] = max(data[i] + bias[i % cols], Type(0));
    }
}

// FC + ReLU 反向的第一步：masked = grad * (output > 0)，同时按列求和得到 bias 的梯度（block 划分同 cudaColumnSum）
template <typename Type>
__global__ void cudaReluMaskColumnSum(const Type* grad, const Type* output, Type* masked, Type* grad_bias,
                                      const int rows, const int cols){
    __shared__ Type partial[8][33];
    const int j = blockIdx.x * 32 + threadIdx.x;
    Type sum = 0;
    if (j < cols){
        for (int i = threadIdx.y; i < rows; i += 8){
            const size_t k = (size_t)i * cols + j;
            const Type g = output[k] > Type(0) ? grad[k] : Type(0);
            masked[k] = g;
            sum += g;
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && j < cols){
        #pragma unroll
        for (int k = 1; k < 8; ++k) sum += partial[k][threadIdx.x];
        grad_bias[j] = sum;
    }
}

// softmax + 交叉熵 + 对 logits 的梯度合并成一个 kernel（单个 block，每个线程处理一行）：
// loss[0] = scale * sum_i (logsumexp(z_i) - z_i[y_i])，grad = scale * (softmax - onehot)；grad 可以为空
template <typename Type>
__global__ void cudaSoftmaxCrossEntropy(const Type* logits, const int* labels, Type* loss, Type* grad,
                                        const int rows, const int cols, const Type scale){
    __shared__ Type partial[BLOCK_SIZE];
    Type sum = 0;
    for (int r = threadIdx.x; r < rows; r += blockDim.x){
        const Type* z = logits + (size_t)r * cols;
        Type m = z[0];
        for (int j = 1; j < cols; ++j) m = max(m, z[j]);
        Type denom = 0;
        for (int j = 0; j < cols; ++j) denom += expf(z[j] - m);
        const int y = labels[r];
        sum += logf(denom) - (z[y] - m);
        if (grad != nullptr){
            const Type inv = scale / denom;
            for (int j = 0; j < cols; ++j){
                grad[(size_t)r * cols + j] = expf(z[j] - m) * inv - (j == y ? scale : Type(0));
            }
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        loss[0] = partial[0] * scale;
    }
}

// ---------------------------------------------------------------- 直接卷积
// KH / KW > 0 时卷积核大小在编译期已知（循环完全展开），等于 0 时从 ConvShape 读取。
// 直接卷积只用于通道数很小的情况（Cin * kh * kw <= 256）：每个线程负责一个空间位置上的 CONV_TILE 个通道，
// 输入 / 输出梯度读一次，在寄存器里给多个通道复用。
constexpr int CONV_TILE = 8;

// 前向：一个线程计算一个输出位置上 CONV_TILE 个输出通道
template <typename Type, int KH, int KW, int TILE>
__global__ void cudaConvDirectForward(const Type* __restrict__ input, const Type* __restrict__ weight,
                                      Type* __restrict__ output, const ConvShape s){
    const int kernel_h = KH > 0 ? KH : s.kernel_h;
    const int kernel_w = KW > 0 ? KW : s.kernel_w;
    const int taps = kernel_h * kernel_w;
    const int plane = s.out_h * s.out_w;
    const int tiles = (s.c_out + TILE - 1) / TILE;
    const int total = s.batch * tiles * plane;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        const int p = i % plane;
        const int co0 = (i / plane) % tiles * TILE;
        const int n = i / plane / tiles;
        const int co_n = min(TILE, s.c_out - co0);
        const int h0 = p / s.out_w * s.stride_h - s.pad_h;
        const int w0 = p % s.out_w * s.stride_w - s.pad_w;
        Type acc[TILE];
        #pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0;
        for (int ci = 0; ci < s.c_in; ++ci){
            const Type* in = input + ((size_t)n * s.c_in + ci) * s.in_h * s.in_w;
            const Type* wt = weight + ((size_t)co0 * s.c_in + ci) * taps;
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh){
                const int h = h0 + kh;
                if (h < 0 || h >= s.in_h) continue;
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw){
                    const int w = w0 + kw;
                    if (w < 0 || w >= s.in_w) continue;
                    const Type v = in[h * s.in_w + w];
                    #pragma unroll
                    for (int t = 0; t < TILE; ++t){
                        if (t < co_n) acc[t] += v * __ldg(wt + (size_t)t * s.c_in * taps + kh * kernel_w + kw);
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < TILE; ++t){
            if (t < co_n) output[((size_t)n * s.c_out + co0 + t) * plane + p] = acc[t];
        }
    }
}

// conv + ReLU + 2x2 最大池化（步长 2）：一个线程计算一个池化输出位置上 CONV_TILE 个通道。
// 卷积结果只留在寄存器里，不写回显存；argmax 记录窗口内最大值的位置（0-3），
// 窗口内全部 <= 0 时记 -1（ReLU 之后梯度为 0）
template <typename Type, int KH, int KW, int TILE>
__global__ void cudaConvReluPoolForward(const Type* __restrict__ input, const Type* __restrict__ weight,
                                        Type* __restrict__ pooled, int* __restrict__ argmax, const ConvShape s,
                                        const int pool_h, const int pool_w){
    const int kernel_h = KH > 0 ? KH : s.kernel_h;
    const int kernel_w = KW > 0 ? KW : s.kernel_w;
    const int taps = kernel_h * kernel_w;
    const int plane = pool_h * pool_w;
    const int tiles = (s.c_out + TILE - 1) / TILE;
    const int total = s.batch * tiles * plane;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        const int p = i % plane;
        const int co0 = (i / plane) % tiles * TILE;
        const int n = i / plane / tiles;
        const int co_n = min(TILE, s.c_out - co0);
        const int ph = p / pool_w;
        const int pw = p % pool_w;
        Type acc[TILE][4];
        #pragma unroll
        for (int t = 0; t < TILE; ++t){
            #pragma unroll
            for (int q = 0; q < 4; ++q) acc[t][q] = 0;
        }
        bool done = false;
        if constexpr (KH > 0 && KW > 0){
            if (s.stride_h == 1 && s.stride_w == 1){
                // 步长为 1：2x2 个输出位置的感受野合起来是 (KH+1)x(KW+1) 的输入块，每个输入只读一次
                const int h0 = 2 * ph - s.pad_h;
                const int w0 = 2 * pw - s.pad_w;
                for (int ci = 0; ci < s.c_in; ++ci){
                    const Type* in = input + ((size_t)n * s.c_in + ci) * s.in_h * s.in_w;
                    const Type* wt = weight + ((size_t)co0 * s.c_in + ci) * taps;
                    Type x[KH + 1][KW + 1];
                    #pragma unroll
                    for (int r = 0; r <= KH; ++r){
                        const int h = h0 + r;
                        #pragma unroll
                        for (int c = 0; c <= KW; ++c){
                            const int w = w0 + c;
                            x[r][c] = (h >= 0 && h < s.in_h && w >= 0 && w < s.in_w) ? in[h * s.in_w + w] : Type(0);
                        }
                    }
                    #pragma unroll
                    for (int t = 0; t < TILE; ++t){
                        if (t < co_n){
                            #pragma unroll
                            for (int kh = 0; kh < KH; ++kh){
                                #pragma unroll
                                for (int kw = 0; kw < KW; ++kw){
                                    const Type wv = __ldg(wt + (size_t)t * s.c_in * taps + kh * KW + kw);
                                    #pragma unroll
                                    for (int q = 0; q < 4; ++q) acc[t][q] += wv * x[(q >> 1) + kh][(q & 1) + kw];
                                }
                            }
                        }
                    }
                }
                done = true;
            }
        }
        for (int ci = 0; ci < s.c_in && !done; ++ci){
            const Type* in = input + ((size_t)n * s.c_in + ci) * s.in_h * s.in_w;
            const Type* wt = weight + ((size_t)co0 * s.c_in + ci) * taps;
            #pragma unroll
            for (int q = 0; q < 4; ++q){
                const int h0 = (2 * ph + (q >> 1)) * s.stride_h - s.pad_h;
                const int w0 = (2 * pw + (q & 1)) * s.stride_w - s.pad_w;
                #pragma unroll
                for (int kh = 0; kh < kernel_h; ++kh){
                    const int h = h0 + kh;
                    if (h < 0 || h >= s.in_h) continue;
                    #pragma unroll
                    for (int kw = 0; kw < kernel_w; ++kw){
                        const int w = w0 + kw;
                        if (w < 0 || w >= s.in_w) continue;
                        const Type v = in[h * s.in_w + w];
                        #pragma unroll
                        for (int t = 0; t < TILE; ++t){
                            if (t < co_n) acc[t][q] += v * __ldg(wt + (size_t)t * s.c_in * taps + kh * kernel_w + kw);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < TILE; ++t){
            if (t < co_n){
                Type best = 0;
                int arg = -1;
                #pragma unroll
                for (int q = 0; q < 4; ++q){
                    if (acc[t][q] > best){
                        best = acc[t][q];
                        arg = q;
                    }
                }
                const size_t o = ((size_t)n * s.c_out + co0 + t) * plane + p;
                pooled[o] = best;
                argmax[o] = arg;
            }
        }
    }
}

// 由池化输出的梯度和 argmax 还原卷积输出（ReLU 之前）的梯度。
// 每个元素只属于一个 2x2 窗口，直接写入，不需要先清零，也不需要 atomicAdd。
// 顺便把紧接着用 atomicAdd 累加的权重梯度 zero_out[0:zero_count] 清零，省掉一次单独的 memset
template <typename Type>
__global__ void cudaReluPoolBackward(const Type* __restrict__ grad_pooled, const int* __restrict__ argmax,
                                     Type* __restrict__ grad_out, const int planes, const int out_h, const int out_w,
                                     const int pool_h, const int pool_w, Type* __restrict__ zero_out, const int zero_count){
    const int total = planes * out_h * out_w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < zero_count; i += blockDim.x * gridDim.x){
        zero_out[i] = 0;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        const int w = i % out_w;
        const int h = (i / out_w) % out_h;
        const int c = i / out_w / out_h;
        const int ph = h >> 1;
        const int pw = w >> 1;
        Type g = 0;
        if (ph < pool_h && pw < pool_w){
            const size_t j = ((size_t)c * pool_h + ph) * pool_w + pw;
            if (argmax[j] == (((h & 1) << 1) | (w & 1))) g = grad_pooled[j];
        }
        grad_out[i] = g;
    }
}

// 输入梯度：一个线程计算一个输入位置上 CONV_TILE 个输入通道，累加所有用到它的输出位置；步长为 1 时省去整除判断
template <typename Type, int KH, int KW, int TILE>
__global__ void cudaConvDirectBackwardInput(const Type* __restrict__ grad_output, const Type* __restrict__ weight,
                                            Type* __restrict__ grad_input, const ConvShape s){
    const int kernel_h = KH > 0 ? KH : s.kernel_h;
    const int kernel_w = KW > 0 ? KW : s.kernel_w;
    const int taps = kernel_h * kernel_w;
    const bool unit_stride = s.stride_h == 1 && s.stride_w == 1;
    const int plane = s.out_h * s.out_w;
    const int in_plane = s.in_h * s.in_w;
    const int tiles = (s.c_in + TILE - 1) / TILE;
    const int total = s.batch * tiles * in_plane;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        const int p = i % in_plane;
        const int ci0 = (i / in_plane) % tiles * TILE;
        const int n = i / in_plane / tiles;
        const int ci_n = min(TILE, s.c_in - ci0);
        const int h = p / s.in_w;
        const int w = p % s.in_w;
        Type acc[TILE];
        #pragma unroll
        for (int t = 0; t < TILE; ++t) acc[t] = 0;
        for (int co = 0; co < s.c_out; ++co){
            const Type* dy = grad_output + ((size_t)n * s.c_out + co) * plane;
            const Type* wt = weight + ((size_t)co * s.c_in + ci0) * taps;
            #pragma unroll
            for (int kh = 0; kh < kernel_h; ++kh){
                int oh = h + s.pad_h - kh;
                if (!unit_stride){
                    if (oh < 0 || oh % s.stride_h != 0) continue;
                    oh /= s.stride_h;
                }
                if (oh < 0 || oh >= s.out_h) continue;
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw){
                    int ow = w + s.pad_w - kw;
                    if (!unit_stride){
                        if (ow < 0 || ow % s.stride_w != 0) continue;
                        ow /= s.stride_w;
                    }
                    if (ow < 0 || ow >= s.out_w) continue;
                    const Type g = dy[oh * s.out_w + ow];
                    #pragma unroll
                    for (int t = 0; t < TILE; ++t){
                        if (t < ci_n) acc[t] += g * __ldg(wt + t * taps + kh * kernel_w + kw);
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < TILE; ++t){
            if (t < ci_n) grad_input[((size_t)n * s.c_in + ci0 + t) * in_plane + p] = acc[t];
        }
    }
}

// 权重梯度（卷积核大小在编译期已知）：grid = (规约分段, Cin, Cout 分块)。
// 每个线程遍历规约区间里的若干输出位置：读一次输入窗口（KH*KW 个值），给 CONV_TILE 个输出通道复用，
// 全部 CONV_TILE * KH * KW 个累加量放在寄存器里；最后 warp shuffle + 共享内存做 block 规约，
// 再 atomicAdd 到 grad_weight（调用前需清零）
template <typename Type, int KH, int KW>
__global__ void cudaConvDirectBackwardWeight(const Type* __restrict__ input, const Type* __restrict__ grad_output,
                                             Type* __restrict__ grad_weight, const ConvShape s){
    constexpr int TAPS = KH * KW;
    constexpr int WARPS = BLOCK_SIZE / 32;
    __shared__ Type warp_sums[CONV_TILE * TAPS][WARPS];
    const int ci = blockIdx.y;
    const int co0 = blockIdx.z * CONV_TILE;
    const int co_n = min(CONV_TILE, s.c_out - co0);
    const int plane = s.out_h * s.out_w;
    const int total = s.batch * plane;
    const int chunk = (total + gridDim.x - 1) / gridDim.x;
    const int begin = blockIdx.x * chunk;
    const int end = min(begin + chunk, total);

    Type acc[CONV_TILE][TAPS];
    #pragma unroll
    for (int t = 0; t < CONV_TILE; ++t){
        #pragma unroll
        for (int k = 0; k < TAPS; ++k) acc[t][k] = 0;
    }
    for (int j = begin + threadIdx.x; j < end; j += blockDim.x){
        const int n = j / plane;
        const int p = j - n * plane;
        const int h0 = p / s.out_w * s.stride_h - s.pad_h;
        const int w0 = p % s.out_w * s.stride_w - s.pad_w;
        const Type* in = input + ((size_t)n * s.c_in + ci) * s.in_h * s.in_w;
        Type x[TAPS];
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh){
            const int h = h0 + kh;
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw){
                const int w = w0 + kw;
                x[kh * KW + kw] = (h >= 0 && h < s.in_h && w >= 0 && w < s.in_w) ? in[h * s.in_w + w] : Type(0);
            }
        }
        const Type* dy = grad_output + ((size_t)n * s.c_out + co0) * plane + p;
        #pragma unroll
        for (int t = 0; t < CONV_TILE; ++t){
            if (t < co_n){
                const Type g = dy[(size_t)t * plane];
                #pragma unroll
                for (int k = 0; k < TAPS; ++k) acc[t][k] += g * x[k];
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int t = 0; t < CONV_TILE; ++t){
        if (t < co_n){
            #pragma unroll
            for (int k = 0; k < TAPS; ++k){
                Type v = acc[t][k];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1){
                    v += __shfl_down_sync(0xffffffff, v, offset);
                }
                if (lane == 0) warp_sums[t * TAPS + k][warp] = v;
            }
        }
    }
    __syncthreads();
    for (int a = threadIdx.x; a < co_n * TAPS; a += blockDim.x){
        Type v = 0;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) v += warp_sums[a][w];
        atomicAdd(grad_weight + ((size_t)(co0 + a / TAPS) * s.c_in + ci) * TAPS + a % TAPS, v);
    }
}

// 任意卷积核大小的权重梯度：每个 block 负责一个权重元素的一段规约
template <typename Type>
__global__ void cudaConvDirectBackwardWeightGeneric(const Type* input, const Type* grad_output, Type* grad_weight, const ConvShape s){
    __shared__ Type partial[BLOCK_SIZE];
    const int kw = blockIdx.x % s.kernel_w;
    const int kh = (blockIdx.x / s.kernel_w) % s.kernel_h;
    const int ci = (blockIdx.x / s.kernel_w / s.kernel_h) % s.c_in;
    const int co = blockIdx.x / s.kernel_w / s.kernel_h / s.c_in;
    const int plane = s.out_h * s.out_w;
    const int total = s.batch * plane;
    const int chunk = (total + gridDim.y - 1) / gridDim.y;
    const int begin = blockIdx.y * chunk;
    const int end = min(begin + chunk, total);
    Type sum = 0;
    for (int j = begin + threadIdx.x; j < end; j += blockDim.x){
        int n = j / plane;
        int oh = (j % plane) / s.out_w;
        int ow = j % s.out_w;
        int h = oh * s.stride_h - s.pad_h + kh;
        int w = ow * s.stride_w - s.pad_w + kw;
        if (h >= 0 && h < s.in_h && w >= 0 && w < s.in_w){
            sum += grad_output[((size_t)n * s.c_out + co) * plane + oh * s.out_w + ow]
                 * input[(((size_t)n * s.c_in + ci) * s.in_h + h) * s.in_w + w];
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicAdd(grad_weight + blockIdx.x, partial[0]);
    }
}

// ---------------------------------------------------------------- 直接卷积的调度
static bool Is3x3(const ConvShape& s){
    return s.kernel_h == 3 && s.kernel_w == 3;
}

static int ConvTiles(int channels){
    return (channels + CONV_TILE - 1) / CONV_TILE;
}

// 每个线程负责几个通道：越大越能复用读进来的数据，但线程数会变少。
// 在线程数不少于约 32K（RTX 4060 Laptop 24 个 SM 大约一整轮）的前提下取最大的分块
static int PickTile(int positions, int channels){
    for (int tile = CONV_TILE; tile > 1; tile /= 2){
        if (tile / 2 >= channels) continue;
        if ((long long)positions * ((channels + tile - 1) / tile) >= 32768) return tile;
    }
    return 1;
}

#define CONV_TILE_SWITCH(tile, CALL) \
    switch (tile){ \
        case 8: CALL(8); break; \
        case 4: CALL(4); break; \
        case 2: CALL(2); break; \
        default: CALL(1); break; \
    }

void ConvDirectForward(const float* input, const float* weight, float* output, const ConvShape& s){
    int positions = s.batch * s.out_h * s.out_w;
    int tile = PickTile(positions, s.c_out);
    int blocks = CudaGetBlocks(positions * ((s.c_out + tile - 1) / tile));
#define LAUNCH(T) \
    if (Is3x3(s)) cudaConvDirectForward<float, 3, 3, T><<<blocks, BLOCK_SIZE>>>(input, weight, output, s); \
    else cudaConvDirectForward<float, 0, 0, T><<<blocks, BLOCK_SIZE>>>(input, weight, output, s);
    CONV_TILE_SWITCH(tile, LAUNCH)
#undef LAUNCH
}

void ConvReluPoolForward(const float* input, const float* weight, float* pooled, int* argmax, const ConvShape& s,
                         int pool_h, int pool_w){
    int positions = s.batch * pool_h * pool_w;
    int tile = PickTile(positions, s.c_out);
    int blocks = CudaGetBlocks(positions * ((s.c_out + tile - 1) / tile));
#define LAUNCH(T) \
    if (Is3x3(s)) cudaConvReluPoolForward<float, 3, 3, T><<<blocks, BLOCK_SIZE>>>(input, weight, pooled, argmax, s, pool_h, pool_w); \
    else cudaConvReluPoolForward<float, 0, 0, T><<<blocks, BLOCK_SIZE>>>(input, weight, pooled, argmax, s, pool_h, pool_w);
    CONV_TILE_SWITCH(tile, LAUNCH)
#undef LAUNCH
}

void ReluPoolBackward(const float* grad_pooled, const int* argmax, float* grad_out, const ConvShape& s,
                      int pool_h, int pool_w, float* zero_out, int zero_count){
    int total = s.batch * s.c_out * s.out_h * s.out_w;
    cudaReluPoolBackward<float><<<CudaGetBlocks(std::max(total, zero_count)), BLOCK_SIZE>>>(
        grad_pooled, argmax, grad_out, s.batch * s.c_out, s.out_h, s.out_w, pool_h, pool_w, zero_out, zero_count);
}

void ConvDirectBackwardInput(const float* grad_output, const float* weight, float* grad_input, const ConvShape& s){
    int positions = s.batch * s.in_h * s.in_w;
    int tile = PickTile(positions, s.c_in);
    int blocks = CudaGetBlocks(positions * ((s.c_in + tile - 1) / tile));
#define LAUNCH(T) \
    if (Is3x3(s)) cudaConvDirectBackwardInput<float, 3, 3, T><<<blocks, BLOCK_SIZE>>>(grad_output, weight, grad_input, s); \
    else cudaConvDirectBackwardInput<float, 0, 0, T><<<blocks, BLOCK_SIZE>>>(grad_output, weight, grad_input, s);
    CONV_TILE_SWITCH(tile, LAUNCH)
#undef LAUNCH
}

void ConvDirectBackwardWeight(const float* input, const float* grad_output, TinyTensor<float>& grad_weight, const ConvShape& s,
                              bool zeroed){
    if (!zeroed) grad_weight.zeros();
    int total = s.batch * s.out_h * s.out_w;
    if (Is3x3(s)){
        // 每个线程至少处理约 8 个输出位置，规约的固定开销才划得来；在此前提下尽量凑够约 48 个 block
        int groups = s.c_in * ConvTiles(s.c_out);
        int max_splits = std::max(1, total / (BLOCK_SIZE * 8));
        int splits = std::max(1, std::min((48 + groups - 1) / groups, max_splits));
        dim3 grid(splits, s.c_in, ConvTiles(s.c_out));
        cudaConvDirectBackwardWeight<float, 3, 3><<<grid, BLOCK_SIZE>>>(input, grad_output, grad_weight.p_data, s);
    }
    else{
        int weights = s.c_out * s.c_in * s.kernel_h * s.kernel_w;
        int max_splits = std::max(1, total / BLOCK_SIZE);
        int splits = std::max(1, std::min((512 + weights - 1) / weights, max_splits));
        cudaConvDirectBackwardWeightGeneric<float><<<dim3(weights, splits), BLOCK_SIZE>>>(input, grad_output, grad_weight.p_data, s);
    }
}

template __global__ void cudaIm2Col<float>(const float*, float*, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int);
template __global__ void cudaCol2Im<float>(const float*, float*, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int);
template __global__ void cudaMaxPoolingForward<float>(const float*, float*, float*, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int, const int);
template __global__ void cudaMaxPoolingBackward<float>(const float*, const float*, float*, const int, const int, const int, const int, const int, const int);
template __global__ void cudaAddBias<float>(float*, const float*, const int, const int, const int);
template __global__ void cudaColumnSum<float>(const float*, float*, const int, const int);
template __global__ void cudaRowSoftmax<float>(const float*, float*, const int, const int);
template __global__ void cudaChannelLog<float>(const float*, float*, const int*, const int, const int);
template __global__ void cudaSoftmaxGrad<float>(const float*, const int*, float*, const int, const int);
template __global__ void cudaSumToScalar<float>(const float*, float*, const int, const float);
template __global__ void cudaAddBiasRelu<float>(float*, const float*, const int, const int);
template __global__ void cudaReluMaskColumnSum<float>(const float*, const float*, float*, float*, const int, const int);
template __global__ void cudaSoftmaxCrossEntropy<float>(const float*, const int*, float*, float*, const int, const int, const float);
