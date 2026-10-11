#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sstream>
#include <cuda_runtime.h>
#include "TinyTensor.h"
#include "TinyTensor_kernels.cuh"

namespace py = pybind11;

template <typename T>
void bind(py::module& m, const std::string& category){
    using Tensor = TinyTensor<T>;
    std::string name = "Tensor_" + category;
    py::class_<Tensor>(m, name.c_str())
        .def(py::init<const std::vector<int>&, const std::string&>())
        .def(py::init<const std::vector<int>&, const std::string&, const std::vector<T>&>())
        .def(py::init<const py::array_t<T>&, const std::string&>())
        .def(py::init<const Tensor&>())
        .def("Size", &Tensor::Size)
        .def("__add__", &Tensor::operator+, py::is_operator())
        .def("__sub__", &Tensor::operator-, py::is_operator())
        .def("__getitem__", [](const Tensor& t, const int k){return t[k];})
        .def("get_shape", &Tensor::get_shape, py::return_value_policy::copy)
        .def("get_device", &Tensor::get_device, py::return_value_policy::copy)
        .def("get_data", &Tensor::get_data, py::return_value_policy::copy)
        .def("flatten", &Tensor::flatten)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("random", &Tensor::random)
        .def("zeros", &Tensor::zeros)
        .def("ones", &Tensor::ones)
        .def("negative", &Tensor::negative)
        .def("mults", &Tensor::mults)
        .def("floatks", &Tensor::floatks)
        .def("resize", &Tensor::resize)
        .def("ReluForward", &Tensor::ReluForward)
        .def("ReluBackward", &Tensor::ReluBackward)
        .def("SigmoidForward", &Tensor::SigmoidForward)
        .def("SigmoidBackward", &Tensor::SigmoidBackward)
        .def("__repr__", [](const Tensor& t){
            std::ostringstream oss;
            oss << t;
            return oss.str();
        });
}

PYBIND11_MODULE(myTensor, m){
    m.doc() = "TinyTensor module";
    bind<float>(m, "float");
    bind<double>(m, "double");
    bind<int>(m, "int");
}