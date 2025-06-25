#ifndef LAYERS_KERNELS_CUH
#define LAYERS_KERNELS_CUH

#include "base.h"
#include "TinyTensor.h"

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
    const int in_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename Type>
void cudaChannelMax(
    const Type* data_in,
    Type* data_out,
    const int kernels_num,
    const int channels
);

template <typename Type>
void cudaChannelSum(
    const Type* data_in,
    Type* data_out,
    const int kernels_num,
    const int channels
);

template <typename Type>
__global__ void cudaChannelSub(
    const Type* data_sub,
    Type* data_out,
    const int kernels_num,
    const int channels
);

template <typename Type>
__global__ void cudaChannelDiv(
    const Type* data_div,
    Type* data_out,
    const int kernels_num,
    const int channels
);

template <typename Type>
__global__ void cudaChannelExp(
    Type* data,
    const int kernels_num
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
__global__ void cudaChannelOne(
    const int* labels,
    Type* values,
    const int kernels_num,
    const int channels
);

//#include "Layers_kernels.inl"

#endif