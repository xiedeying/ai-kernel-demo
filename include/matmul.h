#pragma once

// 一个简单的矩阵乘法算子接口：
// C = A[M x K] * B[K x N]
// 矩阵按 row-major 存储：
//   A[i, k] 存在 A[i * K + k]
//   B[k, j] 存在 B[k * N + j]
//   C[i, j] 存在 C[i * N + j]

// CPU 版本：单线程 for 循环，作为参考实现 + 正确性基准
void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

// GPU 版本 1：朴素 CUDA kernel
//   - 一个线程负责计算 C 中的一个元素 C[row, col]
void matmul_cuda_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

// GPU 版本 2：使用 shared memory + tiling 优化的 CUDA kernel
//   - 一个 block 负责计算 C 的一个小块（tile）
//   - 通过共享内存复用 A 和 B 的子块，减少 global memory 访问
void matmul_cuda_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
