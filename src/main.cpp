#include "matmul.h"
#include "fused_ops.h"  // 融合算子 & softmax 的头文件
#include "conv2d.h"

#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>

// 计算两个向量之间的最大绝对误差
float max_abs_diff(const std::vector<float>& a,
                   const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector size mismatch");
    }
    float maxDiff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxDiff) maxDiff = d;
    }
    return maxDiff;
}

// 简单计算 GFLOPS：2 * M * N * K / time(s) / 1e9
// 矩阵乘法 C = A * B 中，总乘加次数是 2 * M * N * K（乘 + 加）
double compute_gflops(int M, int N, int K, double time_ms) {
    double ops = 2.0 * M * N * K;
    double time_s = time_ms / 1000.0;
    return ops / time_s / 1e9;
}

// 计算 Conv2D 的 FLOPs：
// ops ≈ 2 * N * C_out * H_out * W_out * C_in * K_h * K_w
double compute_conv2d_gflops(
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    double time_ms
) {
    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    double ops = 2.0 * static_cast<double>(N) * C_out
                 * H_out * W_out * C_in * K_h * K_w;
    double time_s = time_ms / 1000.0;
    return ops / time_s / 1e9;
}

// ==========================
// 1. 原始 MatMul benchmark
// ==========================
void run_matmul_benchmark() {
    // 可以改这里调矩阵大小，但 1024 比较有代表性
    int M = 1024;
    int K = 1024;
    int N = 1024;

    std::cout << "==== MatMul Benchmark (CPU vs CUDA naive/tiled) ====\n";
    std::cout << "Matrix size: "
              << "M=" << M << ", K=" << K << ", N=" << N << "\n";

    // 分配 CPU 内存（使用 std::vector 简化管理）
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C_cpu(static_cast<size_t>(M) * N);
    std::vector<float> C_naive(static_cast<size_t>(M) * N);
    std::vector<float> C_tiled(static_cast<size_t>(M) * N);

    // 用固定 seed 的随机数生成器初始化 A、B，保证每次运行数据一致
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // ==========================
    // 1. CPU baseline
    // ==========================
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, N, K);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_ms = std::chrono::duration<double, std::milli>(
                      t_cpu_end - t_cpu_start
                  ).count();

    std::cout << "[CPU] time: " << cpu_ms << " ms\n";
    std::cout << "[CPU] approx GFLOPS: "
              << compute_gflops(M, N, K, cpu_ms)
              << "\n\n";

    // ==========================
    // 2. CUDA naive 版本
    // ==========================
    // 使用 cudaEvent 计时，方便统计 GPU 端耗时（包括 H2D + kernel + D2H）
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive);
    matmul_cuda_naive(A.data(), B.data(), C_naive.data(), M, N, K);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);

    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start_naive, stop_naive);

    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);

    std::cout << "[CUDA naive] total time (H2D+kernel+D2H): "
              << naive_ms << " ms\n";
    std::cout << "[CUDA naive] approx GFLOPS: "
              << compute_gflops(M, N, K, naive_ms)
              << "\n";

    // 和 CPU 结果做数值对比
    float diff_naive = max_abs_diff(C_cpu, C_naive);
    std::cout << "[CUDA naive] max abs diff vs CPU: "
              << diff_naive << "\n\n";

    // ==========================
    // 3. CUDA tiled 版本
    // ==========================
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_tiled);
    matmul_cuda_tiled(A.data(), B.data(), C_tiled.data(), M, N, K);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);

    float tiled_ms = 0.0f;
    cudaEventElapsedTime(&tiled_ms, start_tiled, stop_tiled);

    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);

    std::cout << "[CUDA tiled] total time (H2D+kernel+D2H): "
              << tiled_ms << " ms\n";
    std::cout << "[CUDA tiled] approx GFLOPS: "
              << compute_gflops(M, N, K, tiled_ms)
              << "\n";

    float diff_tiled = max_abs_diff(C_cpu, C_tiled);
    std::cout << "[CUDA tiled] max abs diff vs CPU: "
              << diff_tiled << "\n\n";

    // ==========================
    // 4. 打印加速比 + 检查结果是否通过
    // ==========================
    std::cout << "Speedup vs CPU (naive): "
              << cpu_ms / naive_ms << "x\n";
    std::cout << "Speedup vs CPU (tiled): "
              << cpu_ms / tiled_ms << "x\n";

    const float tol = 1e-3f; // 容忍的最大误差
    if (diff_naive < tol && diff_tiled < tol) {
        std::cout << "Result check: PASSED (tol=" << tol << ")\n";
    } else {
        std::cout << "Result check: FAILED (tol=" << tol << ")\n";
    }
}

// ==========================
// 2. MatMul + Bias + ReLU 融合算子 benchmark
// ==========================
void run_fused_matmul_benchmark() {
    int M = 1024;
    int K = 1024;
    int N = 1024;

    std::cout << "\n==== MatMul + Bias + ReLU Fused Benchmark ====\n";
    std::cout << "Matrix size (fused): "
              << "M=" << M << ", K=" << K << ", N=" << N << "\n";

    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> bias(static_cast<size_t>(N));
    std::vector<float> C_cpu(static_cast<size_t>(M) * N);
    std::vector<float> C_gpu(static_cast<size_t>(M) * N);

    // 用固定 seed 的随机数生成器初始化，保证可复现
    std::mt19937 rng(23456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    // CPU fused
    auto t_f_cpu_start = std::chrono::high_resolution_clock::now();
    matmul_bias_relu_cpu(
        A.data(), B.data(), bias.data(), C_cpu.data(), M, N, K
    );
    auto t_f_cpu_end = std::chrono::high_resolution_clock::now();
    double f_cpu_ms = std::chrono::duration<double, std::milli>(
                          t_f_cpu_end - t_f_cpu_start
                      ).count();

    std::cout << "[CPU fused] time: " << f_cpu_ms << " ms\n";
    std::cout << "[CPU fused] approx GFLOPS: "
              << compute_gflops(M, N, K, f_cpu_ms) << "\n\n";

    // CUDA fused
    cudaEvent_t f_start_gpu, f_stop_gpu;
    cudaEventCreate(&f_start_gpu);
    cudaEventCreate(&f_stop_gpu);

    cudaEventRecord(f_start_gpu);
    matmul_bias_relu_cuda(
        A.data(), B.data(), bias.data(), C_gpu.data(), M, N, K
    );
    cudaEventRecord(f_stop_gpu);
    cudaEventSynchronize(f_stop_gpu);

    float f_gpu_ms = 0.0f;
    cudaEventElapsedTime(&f_gpu_ms, f_start_gpu, f_stop_gpu);

    cudaEventDestroy(f_start_gpu);
    cudaEventDestroy(f_stop_gpu);

    std::cout << "[CUDA fused] total time (H2D+kernel+D2H): "
              << f_gpu_ms << " ms\n";
    std::cout << "[CUDA fused] approx GFLOPS: "
              << compute_gflops(M, N, K, f_gpu_ms) << "\n";

    float diff_fused = max_abs_diff(C_cpu, C_gpu);
    std::cout << "[CUDA fused] max abs diff vs CPU: "
              << diff_fused << "\n";

    // 这里也打印加速比
    std::cout << "Speedup vs CPU (fused): "
              << f_cpu_ms / f_gpu_ms << "x\n";
}

// ==========================
// 3. Softmax row-wise benchmark
// ==========================
void run_softmax_benchmark() {
    std::cout << "\n==== Softmax (row-wise, numerically stable) ====\n";

    int M = 2048; // 行数
    int N = 1024; // 每行长度

    std::cout << "Softmax size: M=" << M
              << ", N=" << N << "\n";

    std::vector<float> logits(static_cast<size_t>(M) * N);
    std::vector<float> probs_cpu(static_cast<size_t>(M) * N);
    std::vector<float> probs_gpu(static_cast<size_t>(M) * N);

    // 同样用固定 seed，保证可复现
    std::mt19937 rng(34567);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : logits) v = dist(rng);

    auto t_soft_cpu_start = std::chrono::high_resolution_clock::now();
    softmax_rowwise_cpu(
        logits.data(), probs_cpu.data(), M, N
    );
    auto t_soft_cpu_end = std::chrono::high_resolution_clock::now();
    double soft_cpu_ms = std::chrono::duration<double, std::milli>(
                             t_soft_cpu_end - t_soft_cpu_start
                         ).count();

    std::cout << "[CPU softmax] time: " << soft_cpu_ms << " ms\n";

    cudaEvent_t start_soft_gpu, stop_soft_gpu;
    cudaEventCreate(&start_soft_gpu);
    cudaEventCreate(&stop_soft_gpu);

    cudaEventRecord(start_soft_gpu);
    softmax_rowwise_cuda(
        logits.data(), probs_gpu.data(), M, N
    );
    cudaEventRecord(stop_soft_gpu);
    cudaEventSynchronize(stop_soft_gpu);

    float soft_gpu_ms = 0.0f;
    cudaEventElapsedTime(&soft_gpu_ms, start_soft_gpu, stop_soft_gpu);

    cudaEventDestroy(start_soft_gpu);
    cudaEventDestroy(stop_soft_gpu);

    std::cout << "[CUDA softmax] total time (H2D+kernel+D2H): "
              << soft_gpu_ms << " ms\n";

    float diff_soft = max_abs_diff(probs_cpu, probs_gpu);
    std::cout << "[CUDA softmax] max abs diff vs CPU: "
              << diff_soft << "\n";

    // 这里同样打印加速比
    std::cout << "Speedup vs CPU (softmax): "
              << soft_cpu_ms / soft_gpu_ms << "x\n";

    // 简单检查每行和是否接近 1（抽查几行）
    int check_rows = std::min(M, 5);
    for (int i = 0; i < check_rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            sum += probs_gpu[i * N + j];
        }
        std::cout << "Row " << i << " sum(prob) ≈ " << sum << "\n";
    }
}

// ==========================
// 4. Conv2D benchmark（NCHW）
// ==========================
void run_conv2d_benchmark() {
    std::cout << "\n==== Conv2D NCHW Benchmark (CPU vs CUDA naive) ====\n";

    // 这里选一个比较典型的卷积配置：
    //   N = 32, C_in = 3, H=W=64, C_out = 16, K = 3x3, stride=1, pad=1
    int N  = 32;
    int C_in  = 3;
    int H_in  = 64;
    int W_in  = 64;
    int C_out = 16;
    int K_h   = 3;
    int K_w   = 3;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h    = 1;
    int pad_w    = 1;

    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    std::cout << "Input:  N=" << N
              << ", C_in=" << C_in
              << ", H_in=" << H_in
              << ", W_in=" << W_in << "\n";
    std::cout << "Filter: C_out=" << C_out
              << ", K_h=" << K_h
              << ", K_w=" << K_w << "\n";
    std::cout << "Output: H_out=" << H_out
              << ", W_out=" << W_out << "\n";

    size_t sizeX = static_cast<size_t>(N) * C_in * H_in * W_in;
    size_t sizeW = static_cast<size_t>(C_out) * C_in * K_h * K_w;
    size_t sizeY = static_cast<size_t>(N) * C_out * H_out * W_out;
    size_t sizeBias = static_cast<size_t>(C_out);

    std::vector<float> X(sizeX);
    std::vector<float> W(sizeW);
    std::vector<float> bias(sizeBias);
    std::vector<float> Y_cpu(sizeY);
    std::vector<float> Y_gpu(sizeY);

    std::mt19937 rng(45678);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : X)    v = dist(rng);
    for (auto& v : W)    v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    // CPU baseline
    auto t_conv_cpu_start = std::chrono::high_resolution_clock::now();
    conv2d_nchw_cpu(
        X.data(), W.data(), bias.data(), Y_cpu.data(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
    auto t_conv_cpu_end = std::chrono::high_resolution_clock::now();
    double conv_cpu_ms = std::chrono::duration<double, std::milli>(
                             t_conv_cpu_end - t_conv_cpu_start
                         ).count();

    std::cout << "[CPU conv2d] time: " << conv_cpu_ms << " ms\n";
    std::cout << "[CPU conv2d] approx GFLOPS: "
              << compute_conv2d_gflops(
                     N, C_in, H_in, W_in,
                     C_out, K_h, K_w,
                     stride_h, stride_w,
                     pad_h, pad_w,
                     conv_cpu_ms
                 ) << "\n\n";

    // CUDA naive conv2d
    cudaEvent_t conv_start_gpu, conv_stop_gpu;
    cudaEventCreate(&conv_start_gpu);
    cudaEventCreate(&conv_stop_gpu);

    cudaEventRecord(conv_start_gpu);
    conv2d_nchw_cuda_naive(
        X.data(), W.data(), bias.data(), Y_gpu.data(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w
    );
    cudaEventRecord(conv_stop_gpu);
    cudaEventSynchronize(conv_stop_gpu);

    float conv_gpu_ms = 0.0f;
    cudaEventElapsedTime(&conv_gpu_ms, conv_start_gpu, conv_stop_gpu);

    cudaEventDestroy(conv_start_gpu);
    cudaEventDestroy(conv_stop_gpu);

    std::cout << "[CUDA conv2d naive] total time (H2D+kernel+D2H): "
              << conv_gpu_ms << " ms\n";
    std::cout << "[CUDA conv2d naive] approx GFLOPS: "
              << compute_conv2d_gflops(
                     N, C_in, H_in, W_in,
                     C_out, K_h, K_w,
                     stride_h, stride_w,
                     pad_h, pad_w,
                     conv_gpu_ms
                 ) << "\n";

    float conv_diff = max_abs_diff(Y_cpu, Y_gpu);
    std::cout << "[CUDA conv2d naive] max abs diff vs CPU: "
              << conv_diff << "\n";
    std::cout << "Speedup vs CPU (conv2d): "
              << conv_cpu_ms / conv_gpu_ms << "x\n";
}

int main() {
    run_matmul_benchmark();
    run_fused_matmul_benchmark();
    run_softmax_benchmark();
    run_conv2d_benchmark(); 
    return 0;
}
