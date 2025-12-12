
# ai-kernel-demo

一个用于练习 **C++/CUDA AI 基础算子开发与优化** 的小型算子库 + benchmark 工程。

当前包含的算子：

- 矩阵乘法 MatMul（CPU / CUDA naive / CUDA tiled）
- 融合算子：MatMul + Bias + ReLU（CPU / CUDA）
- 数值稳定 Softmax（按行 row-wise，CPU / CUDA）
- 2D 卷积 Conv2D（NCHW，CPU / CUDA naive）

并配套：

- 统一的 C++ benchmark 程序（`main.cpp`）
- CMake 构建配置
- 简单的误差校验与 GFLOPS 统计

项目定位：

> 不是为了“造一个框架”，而是用几个典型算子，把 **CUDA 并行模型、shared memory 优化、算子融合、数值稳定性、基准测试** 这些关键点走一遍，方便在面试 / 学习中展开说明。

---

## 1. 环境要求

- C++17 编译器（g++ / clang++ 等）
- CMake ≥ 3.18
- CUDA Toolkit（推荐 11.x+）
- 一块支持 CUDA 的 NVIDIA GPU  
  （开发环境实际测试：RTX 3080 + WSL2 Ubuntu）

---

## 2. 目录结构

```text
ai-kernel-demo/
  CMakeLists.txt
  README.md

  include/
    matmul.h       # MatMul 接口声明
    fused_ops.h    # MatMul + Bias + ReLU & Softmax 接口
    conv2d.h       # 2D 卷积 Conv2D 接口

  src/
    matmul_cpu.cpp # MatMul 的 CPU 实现
    matmul_cuda.cu # MatMul 的 CUDA naive / tiled 实现
    fused_ops.cu   # Fused MatMul + ReLU & Softmax CUDA 实现 + CPU baseline
    conv2d.cu      # Conv2D CPU + CUDA naive 实现
    main.cpp       # 统一 benchmark，运行所有算子


---

## 3. Performance
```text
==== MatMul Benchmark (CPU vs CUDA naive/tiled) ====
Matrix size: M=1024, K=1024, N=1024
[CPU] time: 4045.81 ms
[CPU] approx GFLOPS: 0.530792

[CUDA naive] total time (H2D+kernel+D2H): 11.8662 ms
[CUDA naive] approx GFLOPS: 180.975
[CUDA naive] max abs diff vs CPU: 1.90735e-05

[CUDA tiled] total time (H2D+kernel+D2H): 6.63629 ms
[CUDA tiled] approx GFLOPS: 323.597
[CUDA tiled] max abs diff vs CPU: 1.90735e-05

Speedup vs CPU (naive): 340.953x
Speedup vs CPU (tiled): 609.649x
Result check: PASSED (tol=0.001)

==== MatMul + Bias + ReLU Fused Benchmark ====
Matrix size (fused): M=1024, K=1024, N=1024
[CPU fused] time: 4170.68 ms
[CPU fused] approx GFLOPS: 0.5149

[CUDA fused] total time (H2D+kernel+D2H): 8.05693 ms
[CUDA fused] approx GFLOPS: 266.539
[CUDA fused] max abs diff vs CPU: 1.90735e-05
Speedup vs CPU (fused): 517.652x

==== Softmax (row-wise, numerically stable) ====
Softmax size: M=2048, N=1024
[CPU softmax] time: 31.3469 ms
[CUDA softmax] total time (H2D+kernel+D2H): 5.0471 ms
[CUDA softmax] max abs diff vs CPU: 3.95812e-09
Speedup vs CPU (softmax): 6.21087x
Row 0 sum(prob) ≈ 1
Row 1 sum(prob) ≈ 1
Row 2 sum(prob) ≈ 1
Row 3 sum(prob) ≈ 1
Row 4 sum(prob) ≈ 1

==== Conv2D NCHW Benchmark (CPU vs CUDA naive) ====
Input:  N=32, C_in=3, H_in=64, W_in=64
Filter: C_out=16, K_h=3, K_w=3
Output: H_out=64, W_out=64
[CPU conv2d] time: 318.908 ms
[CPU conv2d] approx GFLOPS: 0.355106

[CUDA conv2d naive] total time (H2D+kernel+D2H): 4.16429 ms
[CUDA conv2d naive] approx GFLOPS: 27.1946
[CUDA conv2d naive] max abs diff vs CPU: 1.90735e-06
Speedup vs CPU (conv2d): 76.5816x
