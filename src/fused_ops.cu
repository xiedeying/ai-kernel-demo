#include "fused_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>

// ================================
// 工具函数：CUDA 错误检查
// ================================
namespace {

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err)
        );
    }
}

} // anonymous namespace

// ================================
// 1. MatMul + Bias + ReLU (CPU)
// ================================
//
// C = ReLU(A * B + bias)
//   A: [M x K]
//   B: [K x N]
//   bias: [N]
//   C: [M x N]
//
void matmul_bias_relu_cpu(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            sum += bias[j];     // 加偏置
            // ReLU 激活：max(0, x)
            C[i * N + j] = sum > 0.0f ? sum : 0.0f;
        }
    }
}

// ================================
// 2. MatMul + Bias + ReLU (CUDA)
// ================================
//
// 一个简单的 fused kernel：
//   - 每个线程计算 C[i, j] 一个元素
//   - 在 kernel 内完成：乘加 + 加 bias + ReLU
//
// 如果你以后想更狠，可以把之前的 tiled MatMul 改成 fused 版本。

__global__
void matmul_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 对应 i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 对应 j

    if (row < M && col < N) {
        float sum = 0.0f;
        // 点乘 A[row, :] 和 B[:, col]
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        sum += bias[col];   // 加列偏置
        // ReLU
        sum = sum > 0.0f ? sum : 0.0f;
        C[row * N + col] = sum;
    }
}

void matmul_bias_relu_cuda(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K
) {
    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeBias = static_cast<size_t>(N) * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dBias = nullptr, *dC = nullptr;

    cudaCheck(cudaMalloc(&dA, sizeA), "cudaMalloc dA failed");
    cudaCheck(cudaMalloc(&dB, sizeB), "cudaMalloc dB failed");
    cudaCheck(cudaMalloc(&dBias, sizeBias), "cudaMalloc dBias failed");
    cudaCheck(cudaMalloc(&dC, sizeC), "cudaMalloc dC failed");

    cudaCheck(cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice),
              "H2D A failed");
    cudaCheck(cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice),
              "H2D B failed");
    cudaCheck(cudaMemcpy(dBias, bias, sizeBias, cudaMemcpyHostToDevice),
              "H2D bias failed");

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    matmul_bias_relu_kernel<<<grid, block>>>(
        dA, dB, dBias, dC, M, N, K
    );
    cudaCheck(cudaGetLastError(), "matmul_bias_relu_kernel launch failed");

    cudaCheck(cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost),
              "D2H C failed");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dBias);
    cudaFree(dC);
}

// ================================
// 3. 数值稳定 Softmax (CPU)
// ================================
//
// 对每一行做 softmax：
//   先减去 max 防止溢出
//   再做 exp 和归一化
//
void softmax_rowwise_cpu(
    const float* logits,
    float* probs,
    int M,
    int N
) {
    for (int i = 0; i < M; ++i) {
        // 1) 找这一行的最大值
        float maxVal = -FLT_MAX;
        for (int j = 0; j < N; ++j) {
            float v = logits[i * N + j];
            if (v > maxVal) maxVal = v;
        }

        // 2) 减去 max，计算 exp，并累加和
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float v = logits[i * N + j] - maxVal;
            float e = std::exp(v);
            probs[i * N + j] = e; // 先把 exp 放这里，后面再除
            sum += e;
        }

        // 3) 除以总和，得到概率
        float invSum = 1.0f / sum;
        for (int j = 0; j < N; ++j) {
            probs[i * N + j] *= invSum;
        }
    }
}

// ================================
// 4. 数值稳定 Softmax (CUDA)
// ================================
//
// 每一行由一个 block 处理：
//   - 先并行求 max
//   - 再并行求 sum(exp(x - max))
//   - 最后归一化
//

__global__
void softmax_rowwise_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int M,
    int N
) {
    // 每个 block 处理一行
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // 共享内存用于 reduce：max & sum
    extern __shared__ float sdata[];

    float* smax = sdata;               // 前半部分存放 max
    float* ssum = sdata + blockSize;   // 后半部分存放 sum

    // ----------------------------
    // 第一步：并行求这一行的最大值
    // ----------------------------
    float localMax = -FLT_MAX;
    for (int j = tid; j < N; j += blockSize) {
        float v = logits[row * N + j];
        if (v > localMax) localMax = v;
    }

    smax[tid] = localMax;
    __syncthreads();

    // 规约求 max
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    float maxVal = smax[0];  // 全行的最大值

    // ----------------------------
    // 第二步：计算 exp(x - max) 并求和
    // ----------------------------
    float localSum = 0.0f;
    for (int j = tid; j < N; j += blockSize) {
        float v = logits[row * N + j] - maxVal;
        float e = expf(v);
        probs[row * N + j] = e;  // 先把 exp 结果存起来
        localSum += e;
    }

    ssum[tid] = localSum;
    __syncthreads();

    // 规约求 sum
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    float sumVal = ssum[0];

    // ----------------------------
    // 第三步：归一化
    // ----------------------------
    float invSum = 1.0f / sumVal;
    for (int j = tid; j < N; j += blockSize) {
        probs[row * N + j] *= invSum;
    }
}

void softmax_rowwise_cuda(
    const float* logits,
    float* probs,
    int M,
    int N
) {
    size_t size = static_cast<size_t>(M) * N * sizeof(float);

    float *dLogits = nullptr, *dProbs = nullptr;
    cudaCheck(cudaMalloc(&dLogits, size), "cudaMalloc dLogits failed");
    cudaCheck(cudaMalloc(&dProbs, size), "cudaMalloc dProbs failed");

    cudaCheck(cudaMemcpy(dLogits, logits, size, cudaMemcpyHostToDevice),
              "H2D logits failed");

    // 每行一个 block，blockDim.x 选择一个合适的值，比如 128 或 256
    int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid(M);

    // 共享内存大小：2 * blockSize * sizeof(float)
    size_t sharedMemBytes = 2 * blockSize * sizeof(float);

    softmax_rowwise_kernel<<<grid, block, sharedMemBytes>>>(
        dLogits, dProbs, M, N
    );
    cudaCheck(cudaGetLastError(), "softmax_rowwise_kernel launch failed");

    cudaCheck(cudaMemcpy(probs, dProbs, size, cudaMemcpyDeviceToHost),
              "D2H probs failed");

    cudaFree(dLogits);
    cudaFree(dProbs);
}
