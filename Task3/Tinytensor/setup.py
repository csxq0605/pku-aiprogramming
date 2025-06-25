import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__ = '0.0.1'
setup(
    name='myTensor',
    version=__version__,
    author='Su Wangjie',
    author_email='wsu0605@pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='myTensor',
            sources=["src/bind_Tensor.cpp", "src/TinyTensor.cu", "src/TinyTensor_kernels.cu"],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '-lcublas']
            })
    ],
    cmdclass={
    'build_ext': BuildExtension
    },
    classifiers=[
    'License :: OSI Approved :: MIT License',
    ],
)

setup(
    name='myLayer',
    version=__version__,
    author='Su Wangjie',
    author_email='wsu0605@pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='myLayer',
            sources=["src/bind_Layer.cpp", "src/Layers.cu", "src/Layers_kernels.cu", "src/TinyTensor.cu", "src/TinyTensor_kernels.cu"],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '-lcublas']
            },
            libraries=['cublas'],  # 链接 cuBLAS 库
            library_dirs=["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"],  # 指定 cuBLAS 库路径
            include_dirs=['C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include']  # 指定 cuBLAS 头文件路径
        )
    ],
    cmdclass={
    'build_ext': BuildExtension
    },
    classifiers=[
    'License :: OSI Approved :: MIT License',
    ],
)