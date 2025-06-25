#ifndef TINY_TENSOR_H
#define TINY_TENSOR_H

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "base.h"

template <typename Type = float>
class TinyTensor{
    public:
        std::vector<int> shape;
        std::string device;
        Type* p_data;
        size_t size;
    
    public:
        TinyTensor(const std::vector<int>& shape, const std::string& device);
        
        TinyTensor(const std::vector<int>& shape, const std::string& device, const std::vector<Type>& data);

        TinyTensor(const pybind11::array_t<Type>& data, const std::string& device);

        TinyTensor(const TinyTensor<Type>& other);

        ~TinyTensor();

        inline size_t Size();

        TinyTensor<Type>& operator=(const TinyTensor<Type>& other);

        TinyTensor<Type> operator+(const TinyTensor<Type>& other) const;

        TinyTensor<Type> operator-(const TinyTensor<Type>& other) const;

        TinyTensor<Type> operator[](const int index) const;

        std::vector<int> get_shape() const;

        std::string get_device() const;

        std::vector<Type> get_data() const;

        TinyTensor<Type> flatten() const;

        TinyTensor<Type>& cpu();

        TinyTensor<Type>& gpu();

        TinyTensor<Type>& random(Type a, Type b);

        TinyTensor<Type>& zeros();

        TinyTensor<Type>& ones();

        TinyTensor<Type>& negative();

        TinyTensor<Type>& mults(const Type scalar);

        TinyTensor<Type>& floatks(float k);

        TinyTensor<Type>& resize(const std::vector<int>& new_shape);

        TinyTensor<Type> ReluForward() const;

        TinyTensor<Type> ReluBackward(const TinyTensor<Type>& grad) const;

        TinyTensor<Type> SigmoidForward() const;

        TinyTensor<Type> SigmoidBackward(const TinyTensor<Type>& grad) const;

        template <typename T>
        friend std::ostream& operator<<(std::ostream& os, const TinyTensor<T>& tensor);
};

//#include "TinyTensor.inl"
#endif