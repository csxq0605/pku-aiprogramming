#ifndef LAYERS_KERNELS_CUH
#define LAYERS_KERNELS_CUH

#include <cublas_v2.h>
#include "base.h"
#include "TinyTensor.h"

// 行主序 GEMM: C[m, n] = alpha * op(A)[m, k] * op(B)[k, n] + beta * C
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
);

// 行主序批量 GEMM，第 i 组矩阵分别偏移 i * stride
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
);

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
);

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
);

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
);

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
);

template <typename Type>
__global__ void cudaAddBias(
    Type* data,
    const Type* bias,
    const int rows,
    const int cols,
    const int bias_size
);

template <typename Type>
__global__ void cudaColumnSum(
    const Type* data,
    Type* out,
    const int rows,
    const int cols
);

template <typename Type>
__global__ void cudaRowSoftmax(
    const Type* data_in,
    Type* data_out,
    const int rows,
    const int cols
);

template <typename Type>
__global__ void cudaChannelLog(
    const Type* data_in,
    Type* data_out,
    const int* labels,
    const int kernels_num,
    const int channels
);

template <typename Type>
__global__ void cudaSoftmaxGrad(
    const Type* softmax,
    const int* labels,
    Type* grad,
    const int rows,
    const int cols
);

// 单个 block 规约：out[0] = scale * sum(data)，结果留在 GPU 上
template <typename Type>
__global__ void cudaSumToScalar(const Type* data, Type* out, const int size, const Type scale);

template <typename Type>
__global__ void cudaAddBiasRelu(Type* data, const Type* bias, const int rows, const int cols);

template <typename Type>
__global__ void cudaReluMaskColumnSum(const Type* grad, const Type* output, Type* masked, Type* grad_bias,
                                      const int rows, const int cols);

template <typename Type>
__global__ void cudaSoftmaxCrossEntropy(const Type* logits, const int* labels, Type* loss, Type* grad,
                                        const int rows, const int cols, const Type scale);

// 直接卷积（通道数较小时比 im2col + GEMM 快）：所有张量为 NCHW，权重为 [Cout, Cin, kh, kw]
struct ConvShape {
    int batch, c_in, c_out, in_h, in_w, out_h, out_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w;
};

void ConvDirectForward(const float* input, const float* weight, float* output, const ConvShape& s);

void ConvDirectBackwardInput(const float* grad_output, const float* weight, float* grad_input, const ConvShape& s);

// conv + ReLU + 2x2 最大池化（步长 2）融合：pooled / argmax 形状为 [N, Cout, pool_h, pool_w]
void ConvReluPoolForward(const float* input, const float* weight, float* pooled, int* argmax, const ConvShape& s,
                         int pool_h, int pool_w);

// 由池化输出的梯度还原卷积输出的梯度（含 ReLU 的梯度），grad_out 形状为 [N, Cout, out_h, out_w]
// 同一个 kernel 顺带把 zero_out[0:zero_count] 清零（给随后的权重梯度累加用）
void ReluPoolBackward(const float* grad_pooled, const int* argmax, float* grad_out, const ConvShape& s,
                      int pool_h, int pool_w, float* zero_out = nullptr, int zero_count = 0);

// zeroed=false 时 grad_weight 会先被清零；调用方已清零时传 true
void ConvDirectBackwardWeight(const float* input, const float* grad_output, TinyTensor<float>& grad_weight, const ConvShape& s,
                              bool zeroed = false);

#endif
