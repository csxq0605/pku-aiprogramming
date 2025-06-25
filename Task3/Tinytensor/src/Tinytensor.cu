#include "base.h"
//#include "Tinytensor_kernels.cuh"
#include "TinyTensor.h"
#include "TinyTensor_kernels.cuh"
#include <algorithm>
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
        cudaMalloc(&p_data, size * sizeof(Type));
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
        Type* c_data = new Type[size];
        std::copy_n(data.begin(), std::min(size, data.size()), c_data);
        cudaMalloc(&p_data, size * sizeof(Type));
        cudaMemcpy(p_data, c_data, size * sizeof(Type), cudaMemcpyHostToDevice);
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::TinyTensor(const pybind11::array_t<Type>& data, const std::string& device) 
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
        cudaMalloc(&p_data, size * sizeof(Type));
        cudaMemcpy(p_data, (Type*)buffer.ptr, size * sizeof(Type), cudaMemcpyHostToDevice);
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
        cudaMalloc(&p_data, size * sizeof(Type));
        cudaMemcpy(p_data, other.p_data, size * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    else{
        throw std::invalid_argument("Invalid device");
    }
}

template <typename Type>
TinyTensor<Type>::~TinyTensor(){
    if (device == "cpu"){
        delete [] p_data;
    }
    else if (device == "gpu"){
        cudaFree(p_data);
    }
}

template <typename Type>
inline size_t TinyTensor<Type>::Size(){
    size_t totalsize = 1;
    for (int dim : shape){
        totalsize *= dim;
    }
    return totalsize;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::operator=(const TinyTensor<Type>& other){
    if (device == "cpu"){
        delete [] p_data;
    }
    else if (device == "gpu"){
        cudaFree(p_data);
    }
    shape = other.shape;
    device = other.device;
    size = Size();
    if (device == "cpu"){
        p_data = new Type[size];
        std::copy(other.p_data, other.p_data + size, p_data);
    }
    else if (device == "gpu"){
        cudaMalloc(&p_data, size * sizeof(Type));
        cudaMemcpy(p_data, other.p_data, size * sizeof(Type), cudaMemcpyDeviceToDevice);
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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        cudaMemcpy(result.p_data, p_data + k * new_size, new_size * sizeof(Type), cudaMemcpyDeviceToDevice);
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
    Type* data = new Type[size];
    if (device == "cpu"){
        std::copy(p_data, p_data + size, data);
    }
    else if (device == "gpu"){
        cudaMemcpy(data, p_data, size * sizeof(Type), cudaMemcpyDeviceToHost);
    }
    std::vector<Type> result(data, data + size);
    delete [] data;
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
        cudaFree(p_data);
        p_data = data;
        device = "cpu";
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::gpu() {
    if (device == "cpu"){
        Type* data = nullptr;
        cudaMalloc(&data, size * sizeof(Type));
        cudaMemcpy(data, p_data, size * sizeof(Type), cudaMemcpyHostToDevice);
        delete [] p_data;
        p_data = data;
        device = "gpu";
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::random(Type a, Type b){
    if (device == "cpu") {
        std::random_device rd;
        std::default_random_engine generator(rd());
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
            cudaRandom<<<CudaGetBlocks(size), BLOCK_SIZE>>>(p_data, size, a, b);
            cudaDeviceSynchronize();
        } else {
            throw "Tensor.random datatype not supported";
        }
        cudaDeviceSynchronize();
    }
    return *this;
}

template <typename Type>
TinyTensor<Type>& TinyTensor<Type>::zeros(){
    if (device == "cpu"){
        std::fill(p_data, p_data + size, 0);
    }
    else if (device == "gpu"){
        cudaMemset(p_data, 0, size * sizeof(Type));
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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
    size = Size();
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
        Type* new_data = nullptr;
        cudaMalloc(&new_data, new_size * sizeof(Type));
        cudaMemcpy(new_data, p_data, std::min(old_size, new_size) * sizeof(Type), cudaMemcpyDeviceToDevice);
        cudaFree(p_data);
        p_data = new_data;
    }
    return *this;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::ReluForward() const{
    TinyTensor<Type> result{*this};
    if (device == "cpu"){
        for (int i = 0; i < result.size; i++){
            result.p_data[i] = (result.p_data[i] > 0) ? result.p_data[i] : 0;
        }
    }
    else if (device == "gpu"){
        Relu_Forward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, result.size);
        cudaDeviceSynchronize();
    }
    return result;
}

template <typename Type>
TinyTensor<Type> TinyTensor<Type>::ReluBackward(const TinyTensor<Type>& grad) const{
    TinyTensor<Type> result{*this};
    if (device == "cpu"){
        if (grad.device == "cpu"){
            for (int i = 0; i < result.size; i++){
                result.p_data[i] = (result.p_data[i] > 0) ? grad.p_data[i] : 0;
            }
        }
        else if (grad.device == "gpu"){
            Type* c_grad = new Type[result.size];
            cudaMemcpy(c_grad, grad.p_data, result.size * sizeof(Type), cudaMemcpyDeviceToHost);
            for (int i = 0; i < result.size; i++){
                result.p_data[i] = (result.p_data[i] > 0) ? c_grad[i] : 0;
            }
            delete [] c_grad;
        }
    }
    else if (device == "gpu"){
        if (grad.device == "cpu"){
            Type* c_grad = nullptr;
            cudaMalloc(&c_grad, result.size * sizeof(Type));
            cudaMemcpy(c_grad, grad.p_data, result.size * sizeof(Type), cudaMemcpyHostToDevice);
            Relu_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, c_grad, result.size);
            cudaDeviceSynchronize();
            cudaFree(c_grad);
        }
        else if (grad.device == "gpu"){
            Relu_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, grad.p_data, result.size);
            cudaDeviceSynchronize();
        }
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
        cudaDeviceSynchronize();
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
            Type* c_grad = nullptr;
            cudaMalloc(&c_grad, result.size * sizeof(Type));
            cudaMemcpy(c_grad, grad.p_data, result.size * sizeof(Type), cudaMemcpyHostToDevice);
            Sigmoid_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, c_grad, result.size);
            cudaDeviceSynchronize();
            cudaFree(c_grad);
        }
        else if (grad.device == "gpu"){
            Sigmoid_Backward<<<CudaGetBlocks(result.size), BLOCK_SIZE>>>(result.p_data, grad.p_data, result.size);
            cudaDeviceSynchronize();
        }
    }
    return result;
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

template class TinyTensor<int>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<int>& tensor);
template class TinyTensor<float>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<float>& tensor);
template class TinyTensor<double>;
template std::ostream& operator<<(std::ostream& os, const TinyTensor<double>& tensor);