#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "base.h"
#include "Layers_kernels.cuh"

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
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, transb, transa, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
    cublasDestroy(handle);
    cudaDeviceSynchronize();
}

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
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        Type* data_col_ptr = data_col + i * kernel_h * kernel_w;
        int index_channel = i % channels;
        int index_kernel = i / channels;
        int index_row = index_kernel / col_w;
        int index_col = index_kernel % col_w;
        int im_row = index_row * stride_h - pad_h;
        int im_col = index_col * stride_w - pad_w;

        for (int row = 0; row < kernel_h; ++row){
            for (int col = 0; col < kernel_w; ++col){
                int row_im = im_row + row;
                int col_im = im_col + col;
                *data_col_ptr = (row_im >= 0 && row_im < im_h && col_im >= 0 && col_im < im_w) ? data_im[(index_channel * im_h + row_im) * im_w + col_im] : 0;
                data_col_ptr++;
            }
        }
    }
}

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
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        int index_w = i % im_w + pad_w;
        int index_h = (i / im_w) % im_h + pad_h;
        int index_channel = i / (im_w * im_h);
        int col_w_start = (index_w < kernel_w) ? 0 : (index_w - kernel_w) / stride_w + 1;
        int col_w_end = min(index_w / stride_w + 1, col_w);
        int col_h_start = (index_h < kernel_h) ? 0 : (index_h - kernel_h) / stride_h + 1;
        int col_h_end = min(index_h / stride_h + 1, col_h);

        Type value = 0;
        for (int h_col = col_h_start; h_col < col_h_end; h_col++){
            for (int w_col = col_w_start; w_col < col_w_end; w_col++){
                int h_im = index_h - h_col * stride_h;
                int w_im = index_w - w_col * stride_w;
                int index_col = (((h_col * col_w + w_col) * channels + index_channel) * kernel_h + h_im) * kernel_w + w_im;
                value += data_col[index_col];
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
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
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
        int max_idx = -1;
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
){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        int index_batch = i / out_h / out_w / channels;
        int index_channel = (i / out_h / out_w) % channels;
        int ph = (i / out_w) % out_h;
        int pw = i % out_w;

        int h_start = ph * stride_h - pad_h;
        int w_start = pw * stride_w - pad_w;
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);

        int index_grad_in = (index_batch * channels + index_channel) * in_h * in_w + mask[i];
        grad_in[index_grad_in] = grad_out[i];
    }
}

template <typename Type>
void cudaChannelMax(
    const Type* data_in,
    Type* data_out,
    const int kernels_num,
    const int channels
){  
    Type* dataout = new Type[kernels_num];
    for (int i = 0; i < kernels_num; i++){
        thrust::device_vector<Type> d_data(data_in + i * channels, data_in + (i + 1) * channels);
        Type max_value = *thrust::max_element(d_data.begin(), d_data.end());
        dataout[i] = max_value;
    }
    cudaMemcpy(data_out, dataout, sizeof(Type) * kernels_num, cudaMemcpyHostToDevice);
}

template <typename Type>
void cudaChannelSum(
    const Type* data_in,
    Type* data_out,
    const int kernels_num,
    const int channels
){
    Type* dataout = new Type[kernels_num];
    for (int i = 0; i < kernels_num; i++){
        thrust::device_vector<Type> d_data(data_in + i * channels, data_in + (i + 1) * channels);
        Type sum_value = thrust::reduce(d_data.begin(), d_data.end());
        dataout[i] = sum_value;
    }
    cudaMemcpy(data_out, dataout, sizeof(Type) * kernels_num, cudaMemcpyHostToDevice);
}

template <typename Type>
__global__ void cudaChannelSub(
    const Type* data_sub,
    Type* data_out,
    const int kernels_num,
    const int channels
){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        for (int j = 0; j < channels; ++j){
            data_out[i * channels + j] -= data_sub[i];
        }
    }
}

template <typename Type>
__global__ void cudaChannelDiv(
    const Type* data_div,
    Type* data_out,
    const int kernels_num,
    const int channels
){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        for (int j = 0; j < channels; ++j){
            data_out[i * channels + j] /= data_div[i];
        }
    }
}

template <typename Type>
__global__ void cudaChannelExp(
    Type* data,
    const int kernels_num
){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        data[i] = expf(data[i]);
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
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        data_out[i] = -log(data_in[i * channels + labels[i]]);
    }
}

template <typename Type>
__global__ void cudaChannelOne(
    const int* labels,
    Type* values,
    const int kernels_num,
    const int channels
){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (kernels_num); i += blockDim.x * gridDim.x){
        values[i * channels + labels[i]] = 1;
    }
}

template __global__ void cudaIm2Col<float>(
    const float* data_im, 
    float* data_col, 
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
template __global__ void cudaCol2Im<float>(
    const float* data_col, 
    float* data_im, 
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
template __global__ void cudaMaxPoolingForward<float>(
    const float* data_in,
    float* data_out,
    float* mask,
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
template __global__ void cudaMaxPoolingBackward<float>(
    const float* grad_out,
    const float* mask,
    float* grad_in,
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
template void cudaChannelMax<float>(
    const float* data_in,
    float* data_out,
    const int kernels_num,
    const int channels
);
template void cudaChannelSum<float>(
    const float* data_in,
    float* data_out,
    const int kernels_num,
    const int channels
);
template __global__ void cudaChannelSub<float>(
    const float* data_sub,
    float* data_out,
    const int kernels_num,
    const int channels
);
template __global__ void cudaChannelDiv<float>(
    const float* data_div,
    float* data_out,
    const int kernels_num,
    const int channels
);
template __global__ void cudaChannelExp<float>(
    float* data,
    const int kernels_num
);
template __global__ void cudaChannelLog<float>(
    const float* data_in,
    float* data_out,
    const int* labels,
    const int kernels_num,
    const int channels
);
template __global__ void cudaChannelOne<float>(
    const int* labels,
    float* values,
    const int kernels_num,
    const int channels
);