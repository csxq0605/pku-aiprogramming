// 全连接层：GEMM 用 cuBLAS，bias + 激活函数（及其反向 + bias 梯度）用一个 kernel 完成
// 行主序：y[rows, out] = x[rows, in] * W[out, in]^T。cuBLAS 是列主序，行主序矩阵等价于它的转置
// FP32 走 cublasSgemm（默认数学模式不使用 TF32），bf16 走 cublasGemmEx（fp32 累加）
#include <cublas_v2.h>
#include <cmath>
#include "nn_common.cuh"
#include "nn_ops.h"

namespace {

cublasHandle_t Handle(){
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr){
        cublasCreate(&handle);
        cublasSetStream(handle, cudaStreamPerThread);
        // 显式给 workspace，CUDA Graph 录制期间 cuBLAS 不会再去分配
        static void* workspace = nullptr;
        const size_t bytes = 32 << 20;
        cudaMalloc(&workspace, bytes);
        cublasSetWorkspace(handle, workspace, bytes);
    }
    return handle;
}

template <typename T> cudaDataType_t CudaType();
template <> cudaDataType_t CudaType<float>(){ return CUDA_R_32F; }
template <> cudaDataType_t CudaType<bf16>(){ return CUDA_R_16BF; }

// 列主序 C(m x n) = alpha * op(A) * op(B) + beta * C
template <typename AB, typename CT>
void Gemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const AB* A, int lda, const AB* B, int ldb,
          CT* C, int ldc, float beta){
    const float alpha = 1.f;
    cublasStatus_t st;
    if (std::is_same<AB, float>::value && std::is_same<CT, float>::value){
        st = cublasSgemm(Handle(), ta, tb, m, n, k, &alpha, (const float*)A, lda, (const float*)B, ldb, &beta, (float*)C, ldc);
    }
    else{
        st = cublasGemmEx(Handle(), ta, tb, m, n, k, &alpha, A, CudaType<AB>(), lda, B, CudaType<AB>(), ldb, &beta,
                          C, CudaType<CT>(), ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    NNCheck(st == CUBLAS_STATUS_SUCCESS, "cublas gemm failed: " + std::to_string((int)st));
}

__device__ __forceinline__ float Gelu(float z){ return 0.5f * z * (1.f + erff(z * 0.70710678f)); }
__device__ __forceinline__ float GeluGrad(float z){
    return 0.5f * (1.f + erff(z * 0.70710678f)) + z * 0.3989422804f * __expf(-0.5f * z * z);
}

// act 0: y += b；1: y = relu(y + b)；2: pre += b, y = gelu(pre)
template <typename T, typename OutT, int ACT>
__global__ void BiasAct(OutT* __restrict__ y, T* __restrict__ pre, const float* __restrict__ b, size_t total, int cols){
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x){
        const float bias = b ? b[i % cols] : 0.f;
        if (ACT == 2){
            float z = ToF(pre[i]) + bias;
            pre[i] = FromF<T>(z);
            y[i] = FromF<OutT>(Gelu(z));
        }
        else{
            float z = ToF(y[i]) + bias;
            y[i] = FromF<OutT>(ACT == 1 ? fmaxf(z, 0.f) : z);
        }
    }
}

// dz = dy * act'(.)，写出 T 类型的 dz，并按列累加得到 bias 梯度。block = 32 列 x 8 行
template <typename T, typename OutT, int ACT>
__global__ void ActBackwardColSum(const OutT* __restrict__ dy, const OutT* __restrict__ y, const T* __restrict__ pre,
                                  T* __restrict__ dz, float* __restrict__ db, int rows, int cols){
    __shared__ float part[8][33];
    const int c = blockIdx.x * 32 + threadIdx.x;
    float s = 0.f;
    if (c < cols){
        for (int r = blockIdx.y * 8 + threadIdx.y; r < rows; r += gridDim.y * 8){
            size_t i = (size_t)r * cols + c;
            float g = ToF(dy[i]);
            if (ACT == 1 && ToF(y[i]) <= 0.f) g = 0.f;
            if (ACT == 2) g *= GeluGrad(ToF(pre[i]));
            if (dz) dz[i] = FromF<T>(g);
            s += g;
        }
    }
    part[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && c < cols && db){
        #pragma unroll
        for (int k = 1; k < 8; ++k) s += part[k][threadIdx.x];
        atomicAdd(db + c, s);
    }
}

int Capped(size_t n){ return std::min(GridFor(n, 256), 24 * 32); }

void Shape2(const std::vector<int>& shape, int& rows, int& cols){
    NNCheck(!shape.empty(), "linear: empty shape");
    cols = shape.back();
    rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
}

}  // namespace

template <typename T, typename OutT>
void LinearForward(const TinyTensor<T>& x, const TinyTensor<T>& w, const TinyTensor<float>* b, TinyTensor<OutT>& y,
                   int act, TinyTensor<T>* pre){
    NNCheckGpu(x, "linear"); NNCheckGpu(w, "linear");
    int rows, in;
    Shape2(x.shape, rows, in);
    NNCheck(w.shape.size() == 2 && w.shape[1] == in, "linear: weight must be [out, in]");
    const int out = w.shape[0];
    std::vector<int> y_shape(x.shape.begin(), x.shape.end() - 1);
    y_shape.push_back(out);
    y.resize_discard(y_shape);
    const float* bias = b ? b->p_data : nullptr;
    if (act == 2){
        NNCheck(pre != nullptr && std::is_same<T, OutT>::value, "linear: gelu needs pre buffer");
        pre->resize_discard(y_shape);
        Gemm<T, T>(CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, w.p_data, in, x.p_data, in, pre->p_data, out, 0.f);
        BiasAct<T, OutT, 2><<<Capped(y.size), 256>>>(y.p_data, pre->p_data, bias, y.size, out);
    }
    else{
        Gemm<T, OutT>(CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, w.p_data, in, x.p_data, in, y.p_data, out, 0.f);
        if (act == 1) BiasAct<T, OutT, 1><<<Capped(y.size), 256>>>(y.p_data, nullptr, bias, y.size, out);
        else if (bias) BiasAct<T, OutT, 0><<<Capped(y.size), 256>>>(y.p_data, nullptr, bias, y.size, out);
    }
}

template <typename T, typename OutT>
void LinearBackward(const TinyTensor<T>& x, const TinyTensor<T>& w, const TinyTensor<OutT>& y, const TinyTensor<OutT>& dy,
                    int act, const TinyTensor<T>* pre, TinyTensor<T>* dx, TinyTensor<float>& dw, TinyTensor<float>* db){
    int rows, in;
    Shape2(x.shape, rows, in);
    const int out = w.shape[0];
    NNCheck(dy.size == (size_t)rows * out, "linear_backward: dy size");
    NNCheck(dw.size == w.size, "linear_backward: dw size");
    // dz：激活函数的反向 + bias 梯度。没有激活且类型相同时直接用 dy
    const T* dz = nullptr;
    TinyTensor<T> dz_buf(std::vector<int>{1}, "gpu");
    const bool need_copy = act != 0 || !std::is_same<T, OutT>::value;
    dim3 grid((out + 31) / 32, std::max(1, std::min(64, rows / 64)));
    if (need_copy){
        dz_buf.resize_discard({rows, out});
        float* dbp = db ? db->p_data : nullptr;
        const T* prep = pre ? pre->p_data : nullptr;
        if (act == 1) ActBackwardColSum<T, OutT, 1><<<grid, dim3(32, 8)>>>(dy.p_data, y.p_data, prep, dz_buf.p_data, dbp, rows, out);
        else if (act == 2) ActBackwardColSum<T, OutT, 2><<<grid, dim3(32, 8)>>>(dy.p_data, y.p_data, prep, dz_buf.p_data, dbp, rows, out);
        else ActBackwardColSum<T, OutT, 0><<<grid, dim3(32, 8)>>>(dy.p_data, y.p_data, prep, dz_buf.p_data, dbp, rows, out);
        dz = dz_buf.p_data;
    }
    else{
        dz = (const T*)dy.p_data;
        if (db) ActBackwardColSum<T, OutT, 0><<<grid, dim3(32, 8)>>>(dy.p_data, y.p_data, nullptr, (T*)nullptr, db->p_data, rows, out);
    }
    // dW[out, in] += dz^T x  ->  列主序 dW^T(in x out) = x^T(in x rows) * dz(rows x out)
    Gemm<T, float>(CUBLAS_OP_N, CUBLAS_OP_T, in, out, rows, x.p_data, in, dz, out, dw.p_data, in, 1.f);
    if (dx){
        // dx[rows, in] = dz W  ->  列主序 dx^T(in x rows) = W^T(in x out) * dz^T(out x rows)
        dx->resize_discard(x.shape);
        Gemm<T, T>(CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out, w.p_data, in, dz, out, dx->p_data, in, 0.f);
    }
}

template void LinearForward<float, float>(const TinyTensor<float>&, const TinyTensor<float>&, const TinyTensor<float>*,
                                          TinyTensor<float>&, int, TinyTensor<float>*);
template void LinearForward<bf16, bf16>(const TinyTensor<bf16>&, const TinyTensor<bf16>&, const TinyTensor<float>*,
                                        TinyTensor<bf16>&, int, TinyTensor<bf16>*);
template void LinearForward<bf16, float>(const TinyTensor<bf16>&, const TinyTensor<bf16>&, const TinyTensor<float>*,
                                         TinyTensor<float>&, int, TinyTensor<bf16>*);
template void LinearBackward<float, float>(const TinyTensor<float>&, const TinyTensor<float>&, const TinyTensor<float>&,
    const TinyTensor<float>&, int, const TinyTensor<float>*, TinyTensor<float>*, TinyTensor<float>&, TinyTensor<float>*);
template void LinearBackward<bf16, bf16>(const TinyTensor<bf16>&, const TinyTensor<bf16>&, const TinyTensor<bf16>&,
    const TinyTensor<bf16>&, int, const TinyTensor<bf16>*, TinyTensor<bf16>*, TinyTensor<float>&, TinyTensor<float>*);
template void LinearBackward<bf16, float>(const TinyTensor<bf16>&, const TinyTensor<bf16>&, const TinyTensor<float>&,
    const TinyTensor<float>&, int, const TinyTensor<bf16>*, TinyTensor<bf16>*, TinyTensor<float>&, TinyTensor<float>*);
