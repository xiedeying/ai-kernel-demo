#pragma once

// 本头文件声明：
// 1. MatMul + Bias + ReLU 融合算子（CPU & CUDA）
// 2. 数值稳定版 Softmax（按行 softmax）（CPU & CUDA）

// ---------------------------
// MatMul + Bias + ReLU
// ---------------------------
//
// 输入：
//   A: [M x K]
//   B: [K x N]
//   bias: [N]       （按列加偏置）
// 输出：
//   C: [M x N]      （C = ReLU(A*B + bias)）
//
// 存储：全部为 row-major
//   A[i, k] = A[i * K + k]
//   B[k, j] = B[k * N + j]
//   C[i, j] = C[i * N + j]

void matmul_bias_relu_cpu(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K
);

void matmul_bias_relu_cuda(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K
);

// ---------------------------
// Row-wise Softmax
// ---------------------------
//
// 输入：
//   logits: [M x N]，每一行做一次 softmax
// 输出：
//   probs:  [M x N]
// 公式：
//   softmax(x_i) = exp(x_i - max) / sum_j exp(x_j - max)
//
// 数值稳定：先减去每行的最大值再做 exp

void softmax_rowwise_cpu(
    const float* logits,
    float* probs,
    int M,
    int N
);

void softmax_rowwise_cuda(
    const float* logits,
    float* probs,
    int M,
    int N
);
