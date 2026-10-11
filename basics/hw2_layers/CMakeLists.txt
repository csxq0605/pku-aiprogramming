cmake_minimum_required(VERSION 3.18)
project(TinyTensorProject LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)

# 查找所有源文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# 添加可执行文件
add_executable(TinyTensorProject ${SOURCES})

# 链接 CUDA 库
find_package(CUDAToolkit REQUIRED)
target_link_libraries(TinyTensorProject CUDA::cudart CUDA::cublas)

# 设置编译选项
set_target_properties(TinyTensorProject PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "all"
)

# 添加 CUDA 编译选项
target_compile_options(TinyTensorProject PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --std=c++17
>)



# 设置字符编码为 UTF-8
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-finput-charset=UTF-8>")
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-finput-charset=UTF-8>")

# 设置输出目录
set_target_properties(TinyTensorProject PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/Debug
)