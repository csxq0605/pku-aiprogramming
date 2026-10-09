#ifndef BASIC_H
#define BASIC_H
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

const int BLOCK_SIZE = 256;

inline int CudaGetBlocks(const int n){
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// 当前线程的默认 stream 是否正在被 CUDA Graph 录制
inline bool IsCapturing(){
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(cudaStreamPerThread, &status);
    return status != cudaStreamCaptureStatusNone;
}

// 简单的显存缓存池：按字节数分桶复用显存，避免每个算子反复 cudaMalloc/cudaFree
// 录制 CUDA Graph 期间分配 / 释放的块进入私有池，只在录制时复用，
// 保证图在重放时使用的显存不会被图外的张量占用
inline std::unordered_map<size_t, std::vector<void*>>& GpuPool(bool capturing = false){
    static std::unordered_map<size_t, std::vector<void*>> pool;
    static std::unordered_map<size_t, std::vector<void*>> graph_pool;
    return capturing ? graph_pool : pool;
}

inline size_t GpuPoolBytes(size_t n, size_t elem){
    size_t bytes = std::max<size_t>(n, 1) * elem;
    return (bytes + 511) / 512 * 512;
}

inline void GpuPoolRelease(){
    for (auto& kv : GpuPool()){
        for (void* p : kv.second){
            cudaFree(p);
        }
    }
    GpuPool().clear();
}

template <typename Type>
inline Type* GpuAlloc(size_t n){
    size_t bytes = GpuPoolBytes(n, sizeof(Type));
    bool capturing = IsCapturing();
    auto it = GpuPool(capturing).find(bytes);
    if (it != GpuPool(capturing).end() && !it->second.empty()){
        void* p = it->second.back();
        it->second.pop_back();
        return (Type*)p;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess && !capturing){
        cudaGetLastError();
        GpuPoolRelease();
        cudaMalloc(&p, bytes);
    }
    return (Type*)p;
}

template <typename Type>
inline void GpuFree(Type* p, size_t n){
    if (p == nullptr){
        return;
    }
    GpuPool(IsCapturing())[GpuPoolBytes(n, sizeof(Type))].push_back((void*)p);
}

// 常驻的锁页内存中转区：新分配的可分页内存直接拷贝到 GPU 很慢，先拷到这里再传输
inline void* PinnedStaging(size_t bytes){
    static void* buffer = nullptr;
    static size_t capacity = 0;
    if (bytes > capacity){
        if (buffer != nullptr){
            cudaFreeHost(buffer);
        }
        capacity = std::max(bytes, capacity * 2);
        cudaMallocHost(&buffer, capacity);
    }
    return buffer;
}

inline void CopyHostToDevice(void* dst, const void* src, size_t bytes){
    if (bytes < (64 << 10)){
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        return;
    }
    void* staging = PinnedStaging(bytes);
    std::copy((const char*)src, (const char*)src + bytes, (char*)staging);
    cudaMemcpy(dst, staging, bytes, cudaMemcpyHostToDevice);
}

inline void CopyDeviceToHost(void* dst, const void* src, size_t bytes){
    if (bytes < (64 << 10)){
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return;
    }
    void* staging = PinnedStaging(bytes);
    cudaMemcpy(staging, src, bytes, cudaMemcpyDeviceToHost);
    std::copy((const char*)staging, (const char*)staging + bytes, (char*)dst);
}

#endif
