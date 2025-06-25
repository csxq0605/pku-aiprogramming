#include <cmath>
#include <cfloat>
#include "base.h"
#include "TinyTensor_kernels.cuh"

template <typename Type>
__global__ void cudaAdd(const Type* data, const Type* other_data, Type* result_data, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        result_data[i] = data[i] + other_data[i];
    }
}

template <typename Type>
__global__ void cudaSub(const Type* data, const Type* other_data, Type* result_data, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        result_data[i] = data[i] - other_data[i];
    }
}

template <typename Type>
__global__ void cudaRandom(Type* data, int size, Type a, Type b){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        curandState state;
        curand_init(clock64(), threadIdx.x, 0, &state);
        float random = curand_uniform(&state);
        data[i] = Type(a + random * (b - a));
    }
}

template <typename Type>
__global__ void cudaOnes(Type* data, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = 1;
    }
}

template <typename Type>
__global__ void cudaNegative(Type* data, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = -data[i];
    }
}

template <typename Type>
__global__ void cudaMults(Type* data, int size, const Type scalar){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] *= scalar;
    }
}

template <typename Type>
__global__ void cudaFloatks(Type* data, int size, float k){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = k;
    }
}

template <typename Type>
__global__ void Relu_Forward(Type* data, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = (data[i] > 0) ? data[i] : 0;
    }
}

template <typename Type>
__global__ void Relu_Backward(Type* data, const Type* grad, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = (data[i] > 0) ? grad[i] : 0;
    }
}

template <typename Type>
__global__ void Sigmoid_Forward(Type* data, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        data[i] = 1 / (1 + expf(-data[i]));
    }
}

template <typename Type>
__global__ void Sigmoid_Backward(Type* data, const Type* grad, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        Type output = 1 / (1 + expf(-data[i]));
        data[i] = output * (1 - output) * grad[i];
    }
}

template __global__ void cudaAdd<float>(const float* data, const float* other_data, float* result_data, int size);
template __global__ void cudaSub<float>(const float* data, const float* other_data, float* result_data, int size);
template __global__ void cudaRandom<float>(float* data, int size, float a, float b);
template __global__ void cudaOnes<float>(float* data, int size);
template __global__ void cudaNegative<float>(float* data, int size);
template __global__ void cudaMults<float>(float* data, int size, const float scalar);
template __global__ void cudaFloatks<float>(float* data, int size, float k);
template __global__ void Relu_Forward<float>(float* data, size_t size);
template __global__ void Relu_Backward<float>(float* data, const float* grad, size_t size);
template __global__ void Sigmoid_Forward<float>(float* data, size_t size);
template __global__ void Sigmoid_Backward<float>(float* data, const float* grad, size_t size);

template __global__ void cudaAdd<int>(const int* data, const int* other_data, int* result_data, int size);
template __global__ void cudaSub<int>(const int* data, const int* other_data, int* result_data, int size);
template __global__ void cudaRandom<int>(int* data, int size, int a, int b);
template __global__ void cudaOnes<int>(int* data, int size);
template __global__ void cudaNegative<int>(int* data, int size);
template __global__ void cudaMults<int>(int* data, int size, const int scalar);
template __global__ void cudaFloatks<int>(int* data, int size, float k);
template __global__ void Relu_Forward<int>(int* data, size_t size);
template __global__ void Relu_Backward<int>(int* data, const int* grad, size_t size);
template __global__ void Sigmoid_Forward<int>(int* data, size_t size);
template __global__ void Sigmoid_Backward<int>(int* data, const int* grad, size_t size);

template __global__ void cudaAdd<double>(const double* data, const double* other_data, double* result_data, int size);
template __global__ void cudaSub<double>(const double* data, const double* other_data, double* result_data, int size);
template __global__ void cudaRandom<double>(double* data, int size, double a, double b);
template __global__ void cudaOnes<double>(double* data, int size);
template __global__ void cudaNegative<double>(double* data, int size);
template __global__ void cudaMults<double>(double* data, int size, const double scalar);
template __global__ void cudaFloatks<double>(double* data, int size, float k);
template __global__ void Relu_Forward<double>(double* data, size_t size);
template __global__ void Relu_Backward<double>(double* data, const double* grad, size_t size);
template __global__ void Sigmoid_Forward<double>(double* data, size_t size);
template __global__ void Sigmoid_Backward<double>(double* data, const double* grad, size_t size);