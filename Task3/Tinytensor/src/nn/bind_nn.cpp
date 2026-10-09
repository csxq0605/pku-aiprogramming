// myNN 模块：bf16 / uint8 张量类型 + 图像模型算子。float / bf16 两个版本用同名重载，pybind 按参数类型分派
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nn_ops.h"

namespace py = pybind11;

template <typename T>
void BindOps(py::module& m){
    m.def("conv_forward", &ConvForward<T>);
    m.def("conv_backward_data", &ConvBackwardData<T>);
    m.def("conv_backward_weight", &ConvBackwardWeight<T>);
    m.def("batchnorm_train", &BatchNormTrain<T>);
    m.def("batchnorm_eval", &BatchNormEval<T>);
    m.def("batchnorm_backward", &BatchNormBackward<T>);
    m.def("layernorm_forward", &LayerNormForward<T>);
    m.def("layernorm_backward", &LayerNormBackward<T>);
    m.def("maxpool2x2_forward", &MaxPool2x2Forward<T>);
    m.def("maxpool2x2_backward", &MaxPool2x2Backward<T>);
    m.def("avgpool_forward", &GlobalAvgPoolForward<T>);
    m.def("avgpool_backward", &GlobalAvgPoolBackward<T>);
    m.def("attention_forward", &AttentionForward<T>);
    m.def("attention_backward", &AttentionBackward<T>);
    m.def("tokens_forward", &TokensForward<T>);
    m.def("tokens_backward", &TokensBackward<T>);
    m.def("select_token_forward", &SelectTokenForward<T>);
    m.def("select_token_backward", &SelectTokenBackward<T>);
    m.def("add", &AddForward<T>);
    m.def("load_batch", &LoadBatch<T>);
}

PYBIND11_MODULE(myNN, m){
    m.doc() = "TinyTensor image-model ops (NHWC, fp32 / bf16)";

    using TB = TinyTensor<bf16>;
    py::class_<TB>(m, "Tensor_bf16")
        .def(py::init<const std::vector<int>&, const std::string&>())
        .def(py::init<const TB&>())
        .def("get_shape", &TB::get_shape)
        .def("get_device", &TB::get_device)
        .def("Size", &TB::Size)
        .def("zeros", &TB::zeros, py::return_value_policy::reference_internal)
        .def("view", &TB::view, py::keep_alive<0, 1>())
        .def("copy_from", &TB::copy_from, py::return_value_policy::reference_internal)
        .def("numpy", [](const TB& t){
            TinyTensor<float> f(std::vector<int>{1}, "gpu");
            CastTensor<bf16, float>(t, f);
            return f.numpy();
        });

    using TU = TinyTensor<unsigned char>;
    py::class_<TU>(m, "Tensor_uint8")
        .def(py::init<const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>&, const std::string&>())
        .def(py::init<const std::vector<int>&, const std::string&>())
        .def("get_shape", &TU::get_shape)
        .def("Size", &TU::Size)
        .def("numpy", &TU::numpy);

    m.def("to_bf16", [](const TinyTensor<float>& src){
        TB dst(std::vector<int>{1}, "gpu");
        CastTensor<float, bf16>(src, dst);
        return dst;
    });
    m.def("cast", &CastTensor<float, bf16>);
    m.def("cast", &CastTensor<bf16, float>);
    m.def("cast", &CastTensor<float, float>);

    BindOps<float>(m);
    BindOps<bf16>(m);

    m.def("linear_forward", &LinearForward<float, float>);
    m.def("linear_forward", &LinearForward<bf16, bf16>);
    m.def("linear_forward", &LinearForward<bf16, float>);
    m.def("linear_backward", &LinearBackward<float, float>);
    m.def("linear_backward", &LinearBackward<bf16, bf16>);
    m.def("linear_backward", &LinearBackward<bf16, float>);

    m.def("softmax_ce", &SoftmaxCrossEntropyForwardBackward);

    py::class_<LrSchedule>(m, "LrSchedule")
        .def(py::init([](float base_lr, int warmup_steps, int total_steps, float final_ratio){
            return LrSchedule{base_lr, warmup_steps, total_steps, final_ratio};
        }), py::arg("base_lr"), py::arg("warmup_steps"), py::arg("total_steps"), py::arg("final_ratio") = 0.f)
        .def_readwrite("base_lr", &LrSchedule::base_lr)
        .def_readwrite("warmup_steps", &LrSchedule::warmup_steps)
        .def_readwrite("total_steps", &LrSchedule::total_steps)
        .def_readwrite("final_ratio", &LrSchedule::final_ratio);
    m.def("mem_info", [](){
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        return std::make_pair(free_bytes, total_bytes);
    }, "(空闲显存, 总显存) 字节数");
    m.def("release_pool", [](){
        cudaDeviceSynchronize();
        GpuPoolRelease();
    }, "把普通显存池里缓存的空闲块还给驱动（CUDA Graph 的私有池不受影响）");
    m.def("sgd_step", &SgdMomentumStep);
    m.def("adamw_step", &AdamWStep);
}
