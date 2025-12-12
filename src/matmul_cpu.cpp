#include "matmul.h"

// 最朴素的三层 for 循环矩阵乘法：
// C[M x N] = A[M x K] * B[K x N]
void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // i: 行索引，遍历 [0, M)
    for (int i = 0; i < M; ++i) {
        // j: 列索引，遍历 [0, N)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            // k: 中间维度 [0, K)，做点乘
            for (int k = 0; k < K; ++k) {
                // A[i, k] * B[k, j]
                sum += A[i * K + k] * B[k * N + j];
            }
            // C[i, j]
            C[i * N + j] = sum;
        }
    }
}
