#include "matmul.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

// 简单的错误检查函数：
// 每次调用 CUDA API 后都用它包一下，方便定位错误。
inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err)
        );
    }
}

// ==========================
// 朴素版本的 CUDA kernel
// ==========================
// __global__ 表示：这是一个在 GPU 上执行、由 CPU 端调用的 kernel 函数。
// 调用方式：kernel<<<gridDim, blockDim>>>(...);
//
// 每个线程负责计算 C 的一个元素：C[row, col]
__global__
void matmul_kernel_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // 当前线程负责的 row / col（二维网格 + 二维线程块）
    // blockIdx: 当前线程块在 grid 中的坐标
    // threadIdx: 当前线程在 block 中的坐标
    // blockDim: 每个 block 的维度大小
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 第几行 i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 第几列 j

    // 越界判断：如果超出矩阵范围就直接退出
    if (row < M && col < N) {
        float sum = 0.0f;
        // K 维度上的累加：点乘
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==========================
// 使用 shared memory 的 tiled 版本
// ==========================
//
// 思路：
// - 一个线程块（block）负责 C 的一个子块（tile）
// - 比如 BLOCK_SIZE = 16，则一个 block 里有 16x16 个线程
//   -> 一次可以算出 C 的 16x16 个元素
// - 把 A 和 B 对应的子块加载到共享内存 As、Bs 中
//   -> 共享内存比 global memory 快很多
//   -> 避免重复访问 global memory，提高性能
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__
void matmul_kernel_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // 声明共享内存（在一个 block 内所有线程共享）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 当前线程负责计算 C 中的哪个元素 (row, col)
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // K 维度被拆成多个 tile，每个 tile 长度为 BLOCK_SIZE
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 遍历所有 tile
    for (int t = 0; t < numTiles; ++t) {
        // 本次 tile 在原始矩阵中的列/行起始位置：
        //   A 的列索引从 t * BLOCK_SIZE 开始
        //   B 的行索引从 t * BLOCK_SIZE 开始
        int tiledColA = t * BLOCK_SIZE + threadIdx.x;
        int tiledRowB = t * BLOCK_SIZE + threadIdx.y;

        // 把 A 的一块加载到共享内存 As 中
        if (row < M && tiledColA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledColA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 把 B 的一块加载到共享内存 Bs 中
        if (tiledRowB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 所有线程都要等 As、Bs 填完
        __syncthreads();

        // 现在可以在这个 tile 内累加贡献了
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 本轮 tile 用完后，再同步一下，避免旧数据影响下一轮
        __syncthreads();
    }

    // 写回结果到全局内存 C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

} // anonymous namespace（内部实现，不暴露给别的翻译单元）

// ==========================
// 下面是 host 端的封装函数（在 CPU 上跑）
// ==========================
//
// 主要负责：
// - 在 GPU 上申请/释放显存
// - 把 A、B 拷过去
// - 启动 __global__ kernel
// - 把 C 拷回来

void matmul_cuda_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // 在 GPU 上分配显存
    cudaCheck(cudaMalloc(&dA, sizeA), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, sizeB), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dC, sizeC), "cudaMalloc dC failed");

    // 把 A、B 从 CPU 拷到 GPU
    cudaCheck(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice),
              "cudaMemcpy A H2D failed");
    cudaCheck(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice),
              "cudaMemcpy B H2D failed");

    // 配置 block 和 grid 大小
    dim3 block(16, 16); // 每个 block 有 16x16 个线程
    dim3 grid(
        (N + block.x - 1) / block.x, // 沿着列方向需要多少个 block
        (M + block.y - 1) / block.y  // 沿着行方向需要多少个 block
    );

    // 启动 kernel（真正的计算在 GPU 上发生）
    matmul_kernel_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaCheck(cudaGetLastError(), "matmul_kernel_naive launch failed");

    // 把结果从 GPU 拷回 CPU
    cudaCheck(cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost),
              "cudaMemcpy C D2H failed");

    // 释放显存
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void matmul_cuda_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    cudaCheck(cudaMalloc(&dA, sizeA), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, sizeB), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dC, sizeC), "cudaMalloc dC failed");

    cudaCheck(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice),
              "cudaMemcpy A H2D failed");
    cudaCheck(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice),
              "cudaMemcpy B H2D failed");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // 启动使用 shared memory 的 kernel
    matmul_kernel_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaCheck(cudaGetLastError(), "matmul_kernel_tiled launch failed");

    cudaCheck(cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost),
              "cudaMemcpy C D2H failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
