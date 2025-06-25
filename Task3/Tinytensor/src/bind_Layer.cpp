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
    m.def("FcBackward", &FcBackward<float>);
    m.def("im2col", &im2col<float>);
    m.def("col2im", &col2im<float>);
    m.def("ConvForward", &ConvForward<float>);
    m.def("ConvBackward", &ConvBackward<float>);
    m.def("MaxPoolingForward", &MaxPoolingForward<float>);
    m.def("MaxPoolingBackward", &MaxPoolingBackward<float>);
    m.def("SoftmaxForward", &SoftmaxForward<float>);
    m.def("SoftmaxLoss", &SoftmaxLoss<float>);
    m.def("CrossEntropyLoss", &CrossEntropyLoss<float>);
    m.def("CrossEntropyLossBackward", &CrossEntropyLossBackward<float>);
}