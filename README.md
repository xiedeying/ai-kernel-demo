
ai-kernel-demo

一个最小可编译的 CUDA AI 算子 Demo：矩阵乘法 MatMul。

包含内容：
matmul_cpu：CPU 参考实现
matmul_cuda_naive：朴素 CUDA kernel，每个线程计算 C 中一个元素
matmul_cuda_tiled：使用 shared memory + tiling 的优化版 CUDA kernel
main.cpp：随机生成矩阵，验证正确性，并做简单 benchmark

构建
mkdir build
cd build
cmake ..
cmake --build . -j

运行
./matmul_bench

输出内容包括：
CPU / GPU 计算时间
naive kernel 与 CPU 结果最大误差
tiled kernel 与 CPU 结果最大误差
简单的 GFLOPS 估算

## Performance
==== MatMul Benchmark (CPU vs CUDA naive/tiled) ====
Matrix size: M=1024, K=1024, N=1024
[CPU] time: 4316.44 ms
[CPU] approx GFLOPS: 0.497512

[CUDA naive] total time (H2D+kernel+D2H): 12.7664 ms
[CUDA naive] approx GFLOPS: 168.213
[CUDA naive] max abs diff vs CPU: 1.90735e-05

[CUDA tiled] total time (H2D+kernel+D2H): 6.67872 ms
[CUDA tiled] approx GFLOPS: 321.541
[CUDA tiled] max abs diff vs CPU: 1.90735e-05

Speedup vs CPU (naive): 338.109x
Speedup vs CPU (tiled): 646.298x
Result check: PASSED (tol=0.001)

==== MatMul + Bias + ReLU Fused Benchmark ====
Matrix size (fused): M=1024, K=1024, N=1024
[CPU fused] time: 4205.72 ms
[CPU fused] approx GFLOPS: 0.51061

[CUDA fused] total time (H2D+kernel+D2H): 8.35178 ms
[CUDA fused] approx GFLOPS: 257.129
[CUDA fused] max abs diff vs CPU: 1.90735e-05
Speedup vs CPU (fused): 503.572x

==== Softmax (row-wise, numerically stable) ====
Softmax size: M=2048, N=1024
[CPU softmax] time: 22.9932 ms
[CUDA softmax] total time (H2D+kernel+D2H): 5.04022 ms
[CUDA softmax] max abs diff vs CPU: 3.95812e-09
Speedup vs CPU (softmax): 4.56194x
Row 0 sum(prob) ≈ 1
Row 1 sum(prob) ≈ 1
Row 2 sum(prob) ≈ 1
Row 3 sum(prob) ≈ 1
Row 4 sum(prob) ≈ 1
