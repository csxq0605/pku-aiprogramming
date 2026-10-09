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
        // numpy 构造放在最前面，避免一维浮点数组在类型转换时被误当成 shape
        .def(py::init<const py::array_t<T, py::array::c_style | py::array::forcecast>&, const std::string&>())
        .def(py::init<const std::vector<int>&, const std::string&>())
        .def(py::init<const std::vector<int>&, const std::string&, const std::vector<T>&>())
        .def(py::init<const Tensor&>())
        .def("Size", &Tensor::Size)
        .def("__add__", &Tensor::operator+, py::is_operator())
        .def("__sub__", &Tensor::operator-, py::is_operator())
        .def("__getitem__", [](const Tensor& t, const int k){return t[k];})
        // 原地修改的方法返回 self，本身不复制（默认策略会把返回的引用复制成新张量）
        .def("get_shape", &Tensor::get_shape, py::return_value_policy::copy)
        .def("get_device", &Tensor::get_device, py::return_value_policy::copy)
        .def("get_data", &Tensor::get_data, py::return_value_policy::copy)
        .def("numpy", &Tensor::numpy)
        .def("__array__", [](const Tensor& t, py::args, py::kwargs){ return t.numpy(); })
        .def("flatten", &Tensor::flatten)
        .def("cpu", &Tensor::cpu, py::return_value_policy::reference_internal)
        .def("gpu", &Tensor::gpu, py::return_value_policy::reference_internal)
        .def("random", &Tensor::random, py::arg("a"), py::arg("b"), py::arg("seed") = -1, py::return_value_policy::reference_internal)
        .def("zeros", &Tensor::zeros, py::return_value_policy::reference_internal)
        .def("ones", &Tensor::ones, py::return_value_policy::reference_internal)
        .def("negative", &Tensor::negative, py::return_value_policy::reference_internal)
        .def("mults", &Tensor::mults, py::return_value_policy::reference_internal)
        .def("floatks", &Tensor::floatks, py::return_value_policy::reference_internal)
        .def("resize", &Tensor::resize, py::return_value_policy::reference_internal)
        .def("ReluForward", &Tensor::ReluForward)
        .def("ReluBackward", &Tensor::ReluBackward)
        .def("SigmoidForward", &Tensor::SigmoidForward)
        .def("SigmoidBackward", &Tensor::SigmoidBackward)
        .def("axpy", &Tensor::axpy, py::return_value_policy::reference_internal)
        .def("adam", &Tensor::adam, py::return_value_policy::reference_internal)
        .def("copy_from", &Tensor::copy_from, py::return_value_policy::reference_internal)
        .def("add_scalar", &Tensor::add_scalar, py::return_value_policy::reference_internal)
        .def("gather_rows", &Tensor::gather_rows, py::return_value_policy::reference_internal)
        .def("gather_batch", &Tensor::gather_batch, py::return_value_policy::reference_internal)
        .def("__repr__", [](const Tensor& t){
            std::ostringstream oss;
            oss << t;
            return oss.str();
        });
}

// 录制得到的 CUDA Graph：之后每次 launch() 重放录制期间提交的全部 GPU 操作
struct CudaGraph{
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~CudaGraph(){
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
    void launch(){
        cudaGraphLaunch(exec, cudaStreamPerThread);
    }
};

static void CheckCuda(cudaError_t err, const char* what){
    if (err != cudaSuccess){
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(myTensor, m){
    m.doc() = "TinyTensor module";
    bind<float>(m, "float");
    bind<double>(m, "double");
    bind<int>(m, "int");
    m.def("synchronize", [](){ cudaDeviceSynchronize(); }, "等待 GPU 上已提交的计算全部完成（用于计时）");
    m.def("axpy_many", [](const std::vector<TinyTensor<float>*>& params, const std::vector<const TinyTensor<float>*>& others,
                          float alpha){
        AxpyMany(params, others, alpha);
    }, "params[i] += alpha * others[i]（一次 kernel 更新多个张量，用于 SGD）");

    m.def("gather_xy", &GatherXYAdvance, "x, y = X[idx], Y[idx]，idx = order[step*stride:][:batch]，然后 step[0] += 1（单个 kernel）");

    py::class_<CudaGraph, std::shared_ptr<CudaGraph>>(m, "CudaGraph")
        .def("launch", &CudaGraph::launch);
    m.def("graph_capture_begin", [](){
        CheckCuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
    }, "开始录制当前线程默认 stream 上的 GPU 操作");
    m.def("graph_capture_end", [](){
        auto g = std::make_shared<CudaGraph>();
        CheckCuda(cudaStreamEndCapture(cudaStreamPerThread, &g->graph), "cudaStreamEndCapture");
        CheckCuda(cudaGraphInstantiate(&g->exec, g->graph, 0), "cudaGraphInstantiate");
        return g;
    }, "结束录制，返回可重放的 CudaGraph");
}