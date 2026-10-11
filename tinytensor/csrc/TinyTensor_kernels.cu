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
__global__ void cudaRandom(Type* data, int size, Type a, Type b, unsigned long long seed){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        // 用全局下标做子序列，保证每个元素的随机数相互独立
        curandState state;
        curand_init(seed, i, 0, &state);
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
__global__ void Relu_Forward(const Type* in, Type* out, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        out[i] = (in[i] > 0) ? in[i] : 0;
    }
}

template <typename Type>
__global__ void Relu_Backward(const Type* in, const Type* grad, Type* out, size_t size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x){
        out[i] = (in[i] > 0) ? grad[i] : 0;
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
template __global__ void cudaRandom<float>(float* data, int size, float a, float b, unsigned long long seed);
template __global__ void cudaOnes<float>(float* data, int size);
template __global__ void cudaNegative<float>(float* data, int size);
template __global__ void cudaMults<float>(float* data, int size, const float scalar);
template __global__ void cudaFloatks<float>(float* data, int size, float k);
template __global__ void Relu_Forward<float>(const float* in, float* out, size_t size);
template __global__ void Relu_Backward<float>(const float* in, const float* grad, float* out, size_t size);
template __global__ void Sigmoid_Forward<float>(float* data, size_t size);
template __global__ void Sigmoid_Backward<float>(float* data, const float* grad, size_t size);

template __global__ void cudaAdd<int>(const int* data, const int* other_data, int* result_data, int size);
template __global__ void cudaSub<int>(const int* data, const int* other_data, int* result_data, int size);
template __global__ void cudaRandom<int>(int* data, int size, int a, int b, unsigned long long seed);
template __global__ void cudaOnes<int>(int* data, int size);
template __global__ void cudaNegative<int>(int* data, int size);
template __global__ void cudaMults<int>(int* data, int size, const int scalar);
template __global__ void cudaFloatks<int>(int* data, int size, float k);
template __global__ void Relu_Forward<int>(const int* in, int* out, size_t size);
template __global__ void Relu_Backward<int>(const int* in, const int* grad, int* out, size_t size);
template __global__ void Sigmoid_Forward<int>(int* data, size_t size);
template __global__ void Sigmoid_Backward<int>(int* data, const int* grad, size_t size);

template __global__ void cudaAdd<double>(const double* data, const double* other_data, double* result_data, int size);
template __global__ void cudaSub<double>(const double* data, const double* other_data, double* result_data, int size);
template __global__ void cudaRandom<double>(double* data, int size, double a, double b, unsigned long long seed);
template __global__ void cudaOnes<double>(double* data, int size);
template __global__ void cudaNegative<double>(double* data, int size);
template __global__ void cudaMults<double>(double* data, int size, const double scalar);
template __global__ void cudaFloatks<double>(double* data, int size, float k);
template __global__ void Relu_Forward<double>(const double* in, double* out, size_t size);
template __global__ void Relu_Backward<double>(const double* in, const double* grad, double* out, size_t size);
template __global__ void Sigmoid_Forward<double>(double* data, size_t size);
template __global__ void Sigmoid_Backward<double>(double* data, const double* grad, size_t size);
__global__ void cudaAxpy(float* y, const float* x, float alpha, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        y[i] += alpha * x[i];
    }
}

__global__ void cudaAdam(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2,
                         float eps, const int* step, int size){
   const float bias_correction1 = 1 - powf(beta1, (float)step[0]);
   const float bias_correction2 = 1 - powf(beta2, (float)step[0]);
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

template <typename Type>
__global__ void cudaAddScalar(Type* data, Type value, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        data[i] += value;
    }
}

template __global__ void cudaAddScalar<float>(float* data, float value, int size);
template __global__ void cudaAddScalar<int>(int* data, int value, int size);
template __global__ void cudaAddScalar<double>(double* data, double value, int size);

template <typename Type>
__global__ void cudaGatherRows(const Type* src, const int* index, Type* out, int row, int size){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        out[i] = src[(size_t)index[i / row] * row + i % row];
    }
}

template __global__ void cudaGatherRows<float>(const float*, const int*, float*, int, int);
template __global__ void cudaGatherRows<int>(const int*, const int*, int*, int, int);
template __global__ void cudaGatherRows<double>(const double*, const int*, double*, int, int);

__global__ void cudaMultiAxpy(const MultiAxpyArgs args){
    const int total = args.start[args.count];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x){
        int k = 0;
        while (i >= args.start[k + 1]) ++k;
        const int j = i - args.start[k];
        args.data[k][j] += args.alpha * args.other[k][j];
    }
}

template <typename Type>
__global__ void cudaGatherBatch(const Type* src, const int* order, const int* step, int stride, Type* out, int row,
                                int size){
    const int* index = order + (size_t)step[0] * stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        out[i] = src[(size_t)index[i / row] * row + i % row];
    }
}
__global__ void cudaGatherXYAdvance(const float* src_x, const int* src_y, const int* order, int* step, int stride,
                                    float* out_x, int* out_y, int row, int batch){
    const int* index = order + (size_t)step[0] * stride;
    const int size = row * batch;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x){
        out_x[i] = src_x[(size_t)index[i / row] * row + i % row];
        if (i < batch) out_y[i] = src_y[index[i]];
    }
    // 最后一个完成的 block 负责推进步数：此时其余 block 都已读过 step[0]
    __syncthreads();
    if (threadIdx.x == 0){
        __threadfence();
        if (atomicAdd(step + 1, 1) == gridDim.x - 1){
            step[0] += 1;
            step[1] = 0;
        }
    }
}

template __global__ void cudaGatherBatch<float>(const float*, const int*, const int*, int, float*, int, int);
template __global__ void cudaGatherBatch<int>(const int*, const int*, const int*, int, int*, int, int);
template __global__ void cudaGatherBatch<double>(const double*, const int*, const int*, int, double*, int, int);
