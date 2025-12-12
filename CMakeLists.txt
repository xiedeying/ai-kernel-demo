cmake_minimum_required(VERSION 3.18)
project(ai_kernel_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可选：根据你的 GPU 架构设置
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_library(matmul
    src/matmul_cpu.cpp
    src/matmul_cuda.cu
    src/fused_ops.cu
)

target_include_directories(matmul PUBLIC include)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(matmul_bench
    src/main.cpp
)

target_link_libraries(matmul PRIVATE CUDA::cudart)
target_link_libraries(matmul_bench PRIVATE matmul CUDA::cudart)
