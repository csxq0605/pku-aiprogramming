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
        // false 表示这是别的张量里的一段（view），析构时不释放显存
        bool owner = true;

    private:
        TinyTensor() : device("gpu"), p_data(nullptr), size(0){}
    
    public:
        TinyTensor(const std::vector<int>& shape, const std::string& device);
        
        TinyTensor(const std::vector<int>& shape, const std::string& device, const std::vector<Type>& data);

        TinyTensor(const pybind11::array_t<Type, pybind11::array::c_style | pybind11::array::forcecast>& data, const std::string& device);

        TinyTensor(const TinyTensor<Type>& other);

        // 移动构造 / 赋值：按值返回张量时直接接管显存，不再整块拷贝
        TinyTensor(TinyTensor<Type>&& other) noexcept;

        TinyTensor<Type>& operator=(TinyTensor<Type>&& other) noexcept;

        ~TinyTensor();

        size_t Size();

        // 不拷贝的视图：从第 offset 个元素开始、形状为 shape；原张量必须比视图活得久
        TinyTensor<Type> view(size_t offset, const std::vector<int>& shape) const;

        TinyTensor<Type>& operator=(const TinyTensor<Type>& other);

        TinyTensor<Type> operator+(const TinyTensor<Type>& other) const;

        TinyTensor<Type> operator-(const TinyTensor<Type>& other) const;

        TinyTensor<Type> operator[](const int index) const;

        std::vector<int> get_shape() const;

        std::string get_device() const;

        std::vector<Type> get_data() const;

        pybind11::array_t<Type> numpy() const;

        TinyTensor<Type> flatten() const;

        TinyTensor<Type>& cpu();

        TinyTensor<Type>& gpu();

        // 均匀分布 (a, b] 随机初始化；seed < 0 时使用随机种子
        TinyTensor<Type>& random(Type a, Type b, long long seed = -1);

        TinyTensor<Type>& zeros();

        TinyTensor<Type>& ones();

        TinyTensor<Type>& negative();

        TinyTensor<Type>& mults(const Type scalar);

        TinyTensor<Type>& floatks(float k);

        TinyTensor<Type>& resize(const std::vector<int>& new_shape);

        // 改变形状；元素个数变化时直接换一块新显存，不保留旧数据（用于马上会被完整写入的输出张量）
        TinyTensor<Type>& resize_discard(const std::vector<int>& new_shape);

        TinyTensor<Type> ReluForward() const;

        TinyTensor<Type> ReluBackward(const TinyTensor<Type>& grad) const;

        TinyTensor<Type> SigmoidForward() const;

        TinyTensor<Type> SigmoidBackward(const TinyTensor<Type>& grad) const;

        // this += alpha * other（用于 SGD 更新与梯度累加）
        TinyTensor<Type>& axpy(const TinyTensor<Type>& other, float alpha);

        // 原地 Adam 更新，m / v 为与参数同形状的状态张量；step 是 GPU 上的步数（从 1 开始）
        TinyTensor<Type>& adam(const TinyTensor<Type>& grad, TinyTensor<Type>& m, TinyTensor<Type>& v,
                               float lr, float beta1, float beta2, float eps, const TinyTensor<int>& step);

        // 从 src 的第 offset 个元素开始拷贝 size 个元素
        TinyTensor<Type>& copy_from(const TinyTensor<Type>& src, size_t offset);

        TinyTensor<Type>& add_scalar(Type value);

        // 按下标取行：this[i] = src[index[i]]（第 0 维为行），用于在 GPU 上打乱数据集
        TinyTensor<Type>& gather_rows(const TinyTensor<Type>& src, const TinyTensor<int>& index);

        // 按 GPU 上的步数取一个 batch：this[i] = src[order[step[0] * stride + i]]（第 0 维为行）
        // 步数和打乱顺序都在 GPU 上，因此连续多个训练 step 可以录进同一张 CUDA Graph
        TinyTensor<Type>& gather_batch(const TinyTensor<Type>& src, const TinyTensor<int>& order,
                                       const TinyTensor<int>& step, int stride);

        template <typename T>
        friend std::ostream& operator<<(std::ostream& os, const TinyTensor<T>& tensor);
};

// params[i] += alpha * others[i]，全部张量在一次（每 MULTI_AXPY_MAX 个一次）kernel 里完成
void AxpyMany(const std::vector<TinyTensor<float>*>& params, const std::vector<const TinyTensor<float>*>& others, float alpha);

// 训练用：x = X[order[step*stride + i]]，y = Y[...]，随后 step[0] += 1；全部在一个 kernel 内完成
// step 至少要有 2 个元素（step[1] 是 kernel 内部使用的完成计数，须初始化为 0）
void GatherXYAdvance(TinyTensor<float>& x, TinyTensor<int>& y, const TinyTensor<float>& src_x, const TinyTensor<int>& src_y,
                     const TinyTensor<int>& order, TinyTensor<int>& step, int stride);

//#include "TinyTensor.inl"
#endif