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
            sources=["src/bind_Tensor.cpp", "src/Tinytensor.cu", "src/TinyTensor_kernels.cu"],
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
            sources=["src/bind_Layer.cpp", "src/Layers.cu", "src/Layers_kernels.cu", "src/Tinytensor.cu", "src/TinyTensor_kernels.cu"],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '-lcublas']
            },
            libraries=['cublas'],  # 链接 cuBLAS 库（库与头文件路径由 CUDAExtension 根据 CUDA_HOME 自动添加）
        )
    ],
    cmdclass={
    'build_ext': BuildExtension
    },
    classifiers=[
    'License :: OSI Approved :: MIT License',
    ],
)