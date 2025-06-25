#ifndef TINY_TENSOR_KERNELS_H
#define TINY_TENSOR_KERNELS_H

#include <curand_kernel.h>

template <typename Type>
__global__ void cudaAdd(const Type* data, const Type* other_data, Type* result_data, int size);

template <typename Type>
__global__ void cudaSub(const Type* data, const Type* other_data, Type* result_data, int size);

template <typename Type>
__global__ void cudaRandom(Type* data, int size, Type a, Type b);

template <typename Type>
__global__ void cudaOnes(Type* data, int size);

template <typename Type>
__global__ void cudaNegative(Type* data, int size);

template <typename Type>
__global__ void cudaMults(Type* data, int size, const Type scalar);

template <typename Type>
__global__ void cudaFloatks(Type* data, int size, float k);

template <typename Type>
__global__ void Relu_Forward(Type* data, size_t size);

template <typename Type>
__global__ void Relu_Backward(Type* data, const Type* grad, size_t size);

template <typename Type>
__global__ void Sigmoid_Forward(Type* data, size_t size);

template <typename Type>
__global__ void Sigmoid_Backward(Type* data, const Type* grad, size_t size);

//#include "TinyTensor_kernels.inl"

#endif