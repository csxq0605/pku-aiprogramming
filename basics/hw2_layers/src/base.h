#ifndef BASE_H
#define BASE_H

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int BLOCK_SIZE = 256;

inline int CudaGetBlocks(const int n){
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

#endif