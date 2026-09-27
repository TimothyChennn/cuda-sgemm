#pragma once
// Every kernel exposes a launcher with this signature.
// Return true once the kernel is implemented; return false to have the
// harness skip it. All matrices are row-major floats.
//   A is M x K, B is K x N, C is M x N.
//   C = alpha * A @ B + beta * C

bool run_k01_naive(int M, int N, int K, float alpha, const float* A,
                   const float* B, float beta, float* C);
bool run_k02_coalesced(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C);
bool run_k03_smem_tiling(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C);
bool run_k04_1d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C);
bool run_k05_2d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C);
bool run_k06_vectorized(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C);

// Helper: integer ceiling division for grid sizing.
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
