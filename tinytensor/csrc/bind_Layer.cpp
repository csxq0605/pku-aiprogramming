#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Layers.h"
#include "Layers_kernels.cuh"

namespace py = pybind11;

PYBIND11_MODULE(myLayer, m){
    m.doc() = "pybind11 myLayer plugin";
    m.def("FcForward", &FcForward<float>);
    m.def("FcBackward", [](const TinyTensor<float>& input, const TinyTensor<float>& output, const TinyTensor<float>& weight,
                           const TinyTensor<float>& bias, TinyTensor<float>& grad_input, const TinyTensor<float>& grad_output,
                           TinyTensor<float>& grad_weight, TinyTensor<float>& grad_bias){
        FcBackward<float>(input, output, weight, bias, grad_input, grad_output, grad_weight, grad_bias, true);
    });
    // 只计算权重与 bias 的梯度（输入不需要梯度时使用）
    m.def("FcBackwardParams", [](const TinyTensor<float>& input, const TinyTensor<float>& weight, const TinyTensor<float>& bias,
                                 const TinyTensor<float>& grad_output, TinyTensor<float>& grad_weight, TinyTensor<float>& grad_bias){
        FcBackward<float>(input, grad_output, weight, bias, grad_weight, grad_output, grad_weight, grad_bias, false);
    });
    m.def("im2col", &im2col<float>);
    m.def("col2im", &col2im<float>);
    m.def("ConvForward", &ConvForward<float>);
    m.def("ConvBackward", [](const TinyTensor<float>& input, const TinyTensor<float>& output, const TinyTensor<float>& weight,
                             TinyTensor<float>& grad_input, const TinyTensor<float>& grad_output, TinyTensor<float>& grad_weight,
                             int pad_h, int pad_w, int stride_h, int stride_w){
        ConvBackward<float>(input, output, weight, grad_input, grad_output, grad_weight, pad_h, pad_w, stride_h, stride_w, true);
    });
    // 只计算卷积核的梯度（输入不需要梯度时使用）
    m.def("ConvBackwardWeight", [](const TinyTensor<float>& input, const TinyTensor<float>& weight, const TinyTensor<float>& grad_output,
                                   TinyTensor<float>& grad_weight, int pad_h, int pad_w, int stride_h, int stride_w){
        ConvBackward<float>(input, grad_output, weight, grad_weight, grad_output, grad_weight, pad_h, pad_w, stride_h, stride_w, false);
    });
    m.def("MaxPoolingForward", &MaxPoolingForward<float>);
    m.def("MaxPoolingBackward", &MaxPoolingBackward<float>);
    m.def("SoftmaxForward", &SoftmaxForward<float>);
    m.def("SoftmaxLoss", &SoftmaxLoss<float>);
    m.def("CrossEntropyLoss", &CrossEntropyLoss<float>);
    m.def("CrossEntropyLossBackward", &CrossEntropyLossBackward<float>);
    // 融合算子
    m.def("FcReluForward", &FcReluForward);
    m.def("FcReluBackward", &FcReluBackward);
    m.def("ConvReluMaxPoolForward", &ConvReluMaxPoolForward);
    m.def("ConvReluMaxPoolBackward", &ConvReluMaxPoolBackward);
    m.def("SoftmaxCrossEntropy", &SoftmaxCrossEntropy);
}