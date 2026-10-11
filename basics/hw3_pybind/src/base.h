#ifndef BASIC_H
#define BASIC_H
#include <cuda_runtime.h>

const int BLOCK_SIZE = 256;

inline int CudaGetBlocks(const int n){
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

#endif