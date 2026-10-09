#include "base.h"
//#include "Tinytensor_kernels.cuh"
#include "TinyTensor.h"
#include "TinyTensor_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

template <typename Type>
TinyTensor<Type>::TinyTensor(const std::vector<int>& shape, const std::string& device) 
    : shape(shape), device(device){
    size = Size();
    if (device == "cpu"){
        p_data = new Type[size];
    }
    else if (device == "gpu"){
        p_data = GpuAlloc<Type>(size);
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::TinyTensor(const std::vector<int>& shape, const std::string& device, const std::vector<Type>& data) 
    : shape(shape), device(device){
    size = Size();
    if (device == "cpu"){
        p_data = new Type[size];
        std::copy_n(data.begin(), std::min(size, data.size()), p_data);
    }
    else if (device == "gpu"){
        std::vector<Type> c_data(size, Type(0));
        std::copy_n(data.begin(), std::min(size, data.size()), c_data.begin());
        p_data = GpuAlloc<Type>(size);
        CopyHostToDevice(p_data, c_data.data(), size * sizeof(Type));
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::TinyTensor(const pybind11::array_t<Type, pybind11::array::c_style | pybind11::array::forcecast>& data, const std::string& device) 
    : device(device){
    auto buffer = data.request();
    size = buffer.size;
    for (auto dim : buffer.shape){
        shape.push_back(dim);
    }
    if (device == "cpu"){
        p_data = new Type[size];
        std::copy_n((Type*)buffer.ptr, size, p_data);
    }
    else if (device == "gpu"){
        p_data = GpuAlloc<Type>(size);
        CopyHostToDevice(p_data, buffer.ptr, size * sizeof(Type));
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::TinyTensor(const TinyTensor<Type>& other)
    : shape(other.shape), device(other.device){
    size = Size();
    if (device == "cpu"){
        p_data = new Type[size];
        std::copy(other.p_data, other.p_data + size, p_data);
    }
    else if (device == "gpu"){
        p_data = GpuAlloc<Type>(size);
        cudaMemcpyAsync(p_data, other.p_data, size * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::TinyTensor(TinyTensor<Type>&& other) noexcept
    : shape(std::move(other.shape)), device(other.device), p_data(other.p_data), size(other.size){
    other.p_data = nullptr;
    other.size = 0;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::operator=(TinyTensor<Type>&& other) noexcept{
    if (this != &other){
        if (device == "cpu"){
            delete [] p_data;
        }
        else if (device == "gpu"){
            GpuFree(p_data, size);
        }
        shape = std::move(other.shape);
        device = other.device;
        p_data = other.p_data;
        size = other.size;
        other.p_data = nullptr;
        other.size = 0;
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>::~TinyTensor(){
    if (device == "cpu"){
        delete [] p_data;
    }
    else if (device == "gpu"){
        GpuFree(p_data, size);
    }
}

template <typename Type>
size_t TinyTensor<Type>::Size(){
    size_t totalsize = 1;
    for (int dim : shape){
        totalsize *= dim;
    }
    return totalsize;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::operator=(const TinyTensor<Type>& other){
    if (this == &other){
        return *this;
    }
    if (device == "cpu"){
        delete [] p_data;
    }
    else if (device == "gpu"){
        GpuFree(p_data, size);
    }
    shape = other.shape;
    device = other.device;
    size = Size();
    if (device == "cpu"){
        p_data = new Type[size];
        std::copy(other.p_data, other.p_data + size, p_data);
    }
    else if (device == "gpu"){
        p_data = GpuAlloc<Type>(size);
        cudaMemcpyAsync(p_data, other.p_data, size * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
    return *this;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::operator+(const TinyTensor<Type>& other) const{
    if (shape != other.shape){
        throw std::invalid_argument("Shape mismatch");
    }
    if (device != other.device){
        throw std::invalid_argument("Device mismatch");
    }
    TinyTensor<Type> result{shape, device};
    result.zeros();
    if (device == "cpu"){
        for (int i = 0; i < size; i++){
            result.p_data[i] = p_data[i] + other.p_data[i];
        }
    }
    else if (device == "gpu"){
        cudaAdd<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, other.p_data, result.p_data, size);
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::operator-(const TinyTensor<Type>& other) const{
    if (shape != other.shape){
        throw std::invalid_argument("Shape mismatch");
    }
    if (device != other.device){
        throw std::invalid_argument("Device mismatch");
    }
    TinyTensor<Type> result{shape, device};
    result.zeros();
    if (device == "cpu"){
        for (int i = 0; i < size; i++){
            result.p_data[i] = p_data[i] - other.p_data[i];
        }
    }
    else if (device == "gpu"){
        cudaSub<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, other.p_data, result.p_data, size);
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::operator[](const int k) const{
    if (k >= shape[0]){
        throw std::invalid_argument("Index out of range");
    }
    size_t new_size = 1;
    std::vector<int> new_shape;
    for (int i = 1; i < shape.size(); i++){
        new_size *= shape[i];
        new_shape.push_back(shape[i]);
    }
    if (new_shape.empty()){
        new_shape.push_back(1);
    }
    
    TinyTensor<Type> result{new_shape, device};
    if (device == "cpu"){
        std::copy(p_data + k * new_size, p_data + (k + 1) * new_size, result.p_data);
    }
    else if (device == "gpu"){
        cudaMemcpyAsync(result.p_data, p_data + k * new_size, new_size * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    return result;
}

template <typename Type>
std::vector<int> TinyTensor<Type>::get_shape() const{
    return shape;
}

template <typename Type>
std::string TinyTensor<Type>::get_device() const{
    return device;
}

template <typename Type>
std::vector<Type> TinyTensor<Type>::get_data() const{
    std::vector<Type> result(size);
    if (device == "cpu"){
        std::copy(p_data, p_data + size, result.begin());
    }
    else if (device == "gpu"){
        CopyDeviceToHost(result.data(), p_data, size * sizeof(Type));
    }
    return result;
}

template <typename Type>
pybind11::array_t<Type> TinyTensor<Type>::numpy() const{
    // 直接拷贝到 numpy 缓冲区，避免经过 Python list
    std::vector<pybind11::ssize_t> np_shape(shape.begin(), shape.end());
    pybind11::array_t<Type> result(np_shape);
    if (device == "cpu"){
        std::copy(p_data, p_data + size, result.mutable_data());
    }
    else if (device == "gpu"){
        CopyDeviceToHost(result.mutable_data(), p_data, size * sizeof(Type));
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::flatten() const{
    int batch_size = 1;
    for (int i = 0; i < shape.size() - 3; i++){
        batch_size *= shape[i];
    }
    TinyTensor<Type> result(*this);
    std::vector<int> new_shape{batch_size, int(result.size) / batch_size};
    result.resize(new_shape);
    return result;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::cpu() {
    if (device == "gpu"){
        Type* data = new Type[size];
        cudaMemcpy(data, p_data, size * sizeof(Type), cudaMemcpyDeviceToHost);
        GpuFree(p_data, size);
        p_data = data;
        device = "cpu";
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::gpu() {
    if (device == "cpu"){
        Type* data = GpuAlloc<Type>(size);
        cudaMemcpy(data, p_data, size * sizeof(Type), cudaMemcpyHostToDevice);
        delete [] p_data;
        p_data = data;
        device = "gpu";
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::random(Type a, Type b, long long seed){
    unsigned long long actual_seed = seed >= 0 ? (unsigned long long)seed : (unsigned long long)std::random_device{}();
    if (device == "cpu") {
        std::default_random_engine generator(actual_seed);
        if constexpr (std::is_floating_point_v<Type>) {
            std::uniform_real_distribution<Type> distribution(a, b);
            for (size_t i = 0; i < size; ++i) {
                p_data[i] = distribution(generator);
            }
        } else if constexpr (std::is_integral_v<Type>) {
            std::uniform_int_distribution<Type> distribution(a, b);
            for (size_t i = 0; i < size; ++i) {
                p_data[i] = distribution(generator);
            }
        } else {
            throw "Tensor.random datatype not supported";
        }
    }
    if (device == "gpu") {
        // kernels from gpu
        if constexpr (std::is_floating_point_v<Type> || std::is_integral_v<Type>) {
            cudaRandom<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size, a, b, actual_seed);
        } else {
            throw "Tensor.random datatype not supported";
        }
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::zeros(){
    if (device == "cpu"){
        std::fill(p_data, p_data + size, 0);
    }
    else if (device == "gpu"){
        cudaMemsetAsync(p_data, 0, size * sizeof(Type));
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::ones(){
    if (device == "cpu"){
        std::fill(p_data, p_data + size, 1);
    }
    else if (device == "gpu"){
        cudaOnes<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size);
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::negative(){
    if (device == "cpu"){
        for (size_t i = 0; i < size; i++){
            p_data[i] = -p_data[i];
        }
    }
    else if (device == "gpu"){
        cudaNegative<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size);
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::mults(const Type scalar){
    if (device == "cpu"){
        for (size_t i = 0; i < size; i++){
            p_data[i] *= scalar;
        }
    }
    else if (device == "gpu"){
        cudaMults<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size, scalar);
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::floatks(float k){
    if (device == "cpu"){
        for (size_t i = 0; i < size; i++){
            p_data[i] = k;
        }
    }
    else if (device == "gpu"){
        cudaFloatks<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size, k);
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::resize(const std::vector<int>& new_shape){
    size_t new_size = 1;
    for (int i = 0; i < new_shape.size(); i++){
        new_size *= new_shape[i];
    }
    size_t old_size = Size();
    shape = new_shape;
    size = new_size;
    if (new_size == old_size){
        return *this;
    }
    if (device == "cpu"){
        Type* new_data = new Type[new_size];
        std::copy(p_data, p_data + std::min(old_size, new_size), new_data);
        delete [] p_data;
        p_data = new_data;
    }
    else if (device == "gpu"){
        Type* new_data = GpuAlloc<Type>(new_size);
        cudaMemcpyAsync(new_data, p_data, std::min(old_size, new_size) * sizeof(Type), cudaMemcpyDeviceToDevice);
        GpuFree(p_data, old_size);
        p_data = new_data;
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::resize_discard(const std::vector<int>& new_shape){
    size_t new_size = 1;
    for (size_t i = 0; i < new_shape.size(); i++){
        new_size *= new_shape[i];
    }
    if (new_size != size){
        if (device == "cpu"){
            delete [] p_data;
            p_data = new Type[new_size];
        }
        else if (device == "gpu"){
            GpuFree(p_data, size);
            p_data = GpuAlloc<Type>(new_size);
        }
    }
    shape = new_shape;
    size = new_size;
    return *this;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::ReluForward() const{
    TinyTensor<Type> result{shape, device};
    if (device == "cpu"){
        for (size_t i = 0; i < size; i++){
            result.p_data[i] = (p_data[i] > 0) ? p_data[i] : 0;
        }
    }
    else if (device == "gpu"){
        Relu_Forward<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, result.p_data, size);
    }
    return result;
}

// 返回 grad * (this > 0)，this 为前向的输入
template <typename Type>
TinyTensor<Type> TinyTensor<Type>::ReluBackward(const TinyTensor<Type>& grad) const{
    if (grad.size != size || grad.device != device){
        throw std::invalid_argument("ReluBackward: shape or device mismatch");
    }
    TinyTensor<Type> result{shape, device};
    if (device == "cpu"){
        for (size_t i = 0; i < size; i++){
            result.p_data[i] = (p_data[i] > 0) ? grad.p_data[i] : 0;
        }
    }
    else if (device == "gpu"){
        Relu_Backward<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, grad.p_data, result.p_data, size);
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::SigmoidForward() const{
    TinyTensor<Type> result{*this};
    if (device == "cpu"){
        for (int i = 0; i < result.size; i++){
            result.p_data[i] = 1 / (1 + exp(-result.p_data[i]));
        }
    }
    else if (device == "gpu"){
        Sigmoid_Forward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, result.size);
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::SigmoidBackward(const TinyTensor<Type>& grad) const{
    TinyTensor<Type> result{*this};
    if (device == "cpu"){
        if (grad.device == "cpu"){
            for (int i = 0; i < result.size; i++){
                Type output = 1 / (1 + exp(-result.p_data[i]));
                result.p_data[i] = output * (1 - output) * grad.p_data[i];
            }
        }
        else if (grad.device == "gpu"){
            Type* c_grad = new Type[result.size];
            cudaMemcpy(c_grad, grad.p_data, result.size * sizeof(Type), cudaMemcpyDeviceToHost);
            for (int i = 0; i < result.size; i++){
                Type output = 1 / (1 + exp(-result.p_data[i]));
                result.p_data[i] = output * (1 - output) * c_grad[i];
            }
            delete [] c_grad;
        }
    }
    else if (device == "gpu"){
        if (grad.device == "cpu"){
            Type* c_grad = GpuAlloc<Type>(result.size);
            cudaMemcpy(c_grad, grad.p_data, result.size * sizeof(Type), cudaMemcpyHostToDevice);
            Sigmoid_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, c_grad, result.size);
            GpuFree(c_grad, result.size);
        }
        else if (grad.device == "gpu"){
            Sigmoid_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, grad.p_data, result.size);
        }
    }
    return result;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::axpy(const TinyTensor<Type>& other, float alpha){
    if (size != other.size || device != "gpu" || other.device != "gpu"){
        throw std::invalid_argument("axpy: shape or device mismatch");
    }
    if constexpr (std::is_same_v<Type, float>){
        cudaAxpy<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, other.p_data, alpha, size);
    }
    else{
        throw std::invalid_argument("axpy only supports float tensors");
    }
    return *this;
}

// 原地 Adam：步数 t 存在 GPU 上（调用前先 add_scalar(1)），这样整个更新可以被 CUDA Graph 录制
template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::adam(const TinyTensor<Type>& grad, TinyTensor<Type>& m, TinyTensor<Type>& v,
                                         float lr, float beta1, float beta2, float eps, const TinyTensor<int>& step){
    if (size != grad.size || size != m.size || size != v.size || device != "gpu" || step.device != "gpu"){
        throw std::invalid_argument("adam: shape or device mismatch");
    }
    if constexpr (std::is_same_v<Type, float>){
        cudaAdam<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, grad.p_data, m.p_data, v.p_data, lr, beta1, beta2,
                                                      eps, step.p_data, size);
    }
    else{
        throw std::invalid_argument("adam only supports float tensors");
    }
    return *this;
}

// 从 src 的第 offset 个元素开始拷贝 size 个元素到本张量（GPU 内拷贝，用于取 batch）
template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::copy_from(const TinyTensor<Type>& src, size_t offset){
    if (offset + size > src.size || device != src.device){
        throw std::invalid_argument("copy_from: out of range or device mismatch");
    }
    if (device == "gpu"){
        cudaMemcpyAsync(p_data, src.p_data + offset, size * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    else{
        std::copy(src.p_data + offset, src.p_data + offset + size, p_data);
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::add_scalar(Type value){
    if (device == "gpu"){
        cudaAddScalar<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, value, size);
    }
    else{
        for (size_t i = 0; i < size; i++){
            p_data[i] += value;
        }
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::gather_rows(const TinyTensor<Type>& src, const TinyTensor<int>& index){
    if (device != "gpu" || src.device != "gpu" || index.device != "gpu" || src.shape.empty()){
        throw std::invalid_argument("gather_rows: tensors must be on gpu");
    }
    size_t row = src.size / src.shape[0];
    if (index.size * row != size){
        throw std::invalid_argument("gather_rows: shape mismatch");
    }
    cudaGatherRows<<<CudaGetBlocks(size), BLOCK_SIZE>>>(src.p_data, index.p_data, p_data, (int)row, (int)size);
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::gather_batch(const TinyTensor<Type>& src, const TinyTensor<int>& order,
                                                 const TinyTensor<int>& step, int stride){
    if (device != "gpu" || src.device != "gpu" || order.device != "gpu" || step.device != "gpu" || src.shape.empty()
        || shape.empty()){
        throw std::invalid_argument("gather_batch: tensors must be on gpu");
    }
    size_t row = src.size / src.shape[0];
    if (shape[0] * row != size || (size_t)stride < shape[0]){
        throw std::invalid_argument("gather_batch: shape mismatch");
    }
    cudaGatherBatch<<<CudaGetBlocks(size), BLOCK_SIZE>>>(src.p_data, order.p_data, step.p_data, stride, p_data,
                                                         (int)row, (int)size);
    return *this;
}

template <typename Type>
std::ostream& operator<<(std::ostream& os, const TinyTensor<Type>& tensor){
    os << "TinyTensor<" << typeid(Type).name() << ">\n";
    os << "Device: " << tensor.device << "\nShape: [";
    for (size_t i = 0; i < tensor.shape.size(); ++i){
        os << tensor.shape[i];
        if (i < tensor.shape.size() - 1){
            os << ", ";
        }
    }
    os << "]\n";
    size_t size = tensor.size;
    if (tensor.device == "cpu"){
        for (size_t i = 0; i < size; ++i) {
            os << tensor.p_data[i] << " ";
            if ((i + 1) % tensor.shape.back() == 0) {
                os << "\n";
            }
        }
    }
    else if (tensor.device == "gpu"){
        TinyTensor<Type> temp_tensor(tensor);
        temp_tensor.cpu();
        for (size_t i = 0; i < temp_tensor.size; ++i) {
            os << temp_tensor.p_data[i] << " ";
            if ((i + 1) % temp_tensor.shape.back() == 0) {
                os << "\n";
            }
        }
    }
    return os;
}

void AxpyMany(const std::vector<TinyTensor<float>*>& params, const std::vector<const TinyTensor<float>*>& others, float alpha){
    if (params.size() != others.size()){
        throw std::invalid_argument("axpy_many: length mismatch");
    }
    for (size_t first = 0; first < params.size(); first += MULTI_AXPY_MAX){
        MultiAxpyArgs args{};
        args.count = (int)std::min<size_t>(MULTI_AXPY_MAX, params.size() - first);
        args.alpha = alpha;
        args.start[0] = 0;
        for (int k = 0; k < args.count; ++k){
            TinyTensor<float>* p = params[first + k];
            const TinyTensor<float>* o = others[first + k];
            if (p->size != o->size || p->device != "gpu" || o->device != "gpu"){
                throw std::invalid_argument("axpy_many: shape or device mismatch");
            }
            args.data[k] = p->p_data;
            args.other[k] = o->p_data;
            args.start[k + 1] = args.start[k] + (int)p->size;
        }
        cudaMultiAxpy<<<CudaGetBlocks(args.start[args.count]), BLOCK_SIZE>>>(args);
    }
}

void GatherXYAdvance(TinyTensor<float>& x, TinyTensor<int>& y, const TinyTensor<float>& src_x, const TinyTensor<int>& src_y,
                     const TinyTensor<int>& order, TinyTensor<int>& step, int stride){
    for (auto device : {x.device, y.device, src_x.device, src_y.device, order.device, step.device}){
        if (device != "gpu") throw std::invalid_argument("gather_xy: tensors must be on gpu");
    }
    if (src_x.shape.empty() || x.shape.empty() || step.size < 2){
        throw std::invalid_argument("gather_xy: bad shape");
    }
    const size_t row = src_x.size / src_x.shape[0];
    const int batch = x.shape[0];
    if (batch * row != x.size || (size_t)batch != y.size || stride < batch){
        throw std::invalid_argument("gather_xy: shape mismatch");
    }
    cudaGatherXYAdvance<<<CudaGetBlocks((int)x.size), BLOCK_SIZE>>>(src_x.p_data, src_y.p_data, order.p_data, step.p_data,
                                                                    stride, x.p_data, y.p_data, (int)row, batch);
}

template class TinyTensor<int>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<int>& tensor);
template class TinyTensor<float>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<float>& tensor);
template class TinyTensor<double>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<double>& tensor);