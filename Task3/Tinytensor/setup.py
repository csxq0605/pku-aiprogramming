"""
用 PyTorch 的 CUDAExtension 编译 myTensor / myLayer（Windows 需要 MSVC，Linux 也可以直接用 build_linux.sh）
用法: python setup.py build_ext --inplace，然后把生成的 myTensor*/myLayer* 放到 optimizer/ 下
"""
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.0.1'
here = os.path.dirname(os.path.abspath(__file__))
pybind11_include = os.path.join(here, "pybind11", "include")

cxx_flags = (['/std:c++17', '/O2', '/DCUDA_API_PER_THREAD_DEFAULT_STREAM'] if sys.platform == 'win32'
             else ['-std=c++17', '-O3', '-DCUDA_API_PER_THREAD_DEFAULT_STREAM'])
nvcc_flags = ['-O3', '-std=c++17', '--default-stream', 'per-thread', '-diag-suppress', '20281']
tensor_sources = ["src/Tinytensor.cu", "src/TinyTensor_kernels.cu"]

setup(
    name='tinytensor',
    version=__version__,
    author='Su Wangjie',
    author_email='wsu0605@pku.edu.cn',
    zip_safe=False,
    install_requires=['torch', 'numpy'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='myTensor',
            sources=["src/bind_Tensor.cpp"] + tensor_sources,
            include_dirs=[pybind11_include],
            extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
        ),
        CUDAExtension(
            name='myLayer',
            sources=["src/bind_Layer.cpp", "src/Layers.cu", "src/Layers_kernels.cu"] + tensor_sources,
            include_dirs=[pybind11_include],
            libraries=['cublas'],
            extra_compile_args={'cxx': cxx_flags, 'nvcc': nvcc_flags},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)
