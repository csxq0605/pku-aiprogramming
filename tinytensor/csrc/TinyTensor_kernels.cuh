#ifndef TINY_TENSOR_KERNELS_H
#define TINY_TENSOR_KERNELS_H

#include <curand_kernel.h>

template <typename Type>
__global__ void cudaAdd(const Type* data, const Type* other_data, Type* result_data, int size);

template <typename Type>
__global__ void cudaSub(const Type* data, const Type* other_data, Type* result_data, int size);

template <typename Type>
__global__ void cudaRandom(Type* data, int size, Type a, Type b, unsigned long long seed);

template <typename Type>
__global__ void cudaOnes(Type* data, int size);

template <typename Type>
__global__ void cudaNegative(Type* data, int size);

template <typename Type>
__global__ void cudaMults(Type* data, int size, const Type scalar);

template <typename Type>
__global__ void cudaFloatks(Type* data, int size, float k);

template <typename Type>
__global__ void Relu_Forward(const Type* in, Type* out, size_t size);

template <typename Type>
__global__ void Relu_Backward(const Type* in, const Type* grad, Type* out, size_t size);

template <typename Type>
__global__ void Sigmoid_Forward(Type* data, size_t size);

template <typename Type>
__global__ void Sigmoid_Backward(Type* data, const Type* grad, size_t size);

__global__ void cudaAxpy(float* y, const float* x, float alpha, int size);

__global__ void cudaAdam(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2,
                         float eps, const int* step, int size);

template <typename Type>
__global__ void cudaAddScalar(Type* data, Type value, int size);

template <typename Type>
__global__ void cudaGatherRows(const Type* src, const int* index, Type* out, int row, int size);

// 一次 kernel 对最多 MULTI_AXPY_MAX 个张量做 data += alpha * other（参数按值传入 kernel，可被 CUDA Graph 录制）
constexpr int MULTI_AXPY_MAX = 8;
struct MultiAxpyArgs {
    float* data[MULTI_AXPY_MAX];
    const float* other[MULTI_AXPY_MAX];
    int start[MULTI_AXPY_MAX + 1];
    int count;
    float alpha;
};

__global__ void cudaMultiAxpy(const MultiAxpyArgs args);

template <typename Type>
__global__ void cudaGatherBatch(const Type* src, const int* order, const int* step, int stride, Type* out, int row,
                                int size);

// 一个 kernel 同时取出 x、y 两个 batch，并在所有 block 读完步数后把 step[0] 加 1（step[1] 为完成计数）
__global__ void cudaGatherXYAdvance(const float* src_x, const int* src_y, const int* order, int* step, int stride,
                                    float* out_x, int* out_y, int row, int batch);

//#include "TinyTensor_kernels.inl"

#endif