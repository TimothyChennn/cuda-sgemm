// Kernel 1: naive.
// One thread per output element. Each thread walks the full K dimension.
// This is the worked example; the other five are yours.
//
// Why it's slow: every thread reads a full row of A and a full column of B
// from global memory, and nothing is reused between threads. Also, look at
// which threads in a warp access which addresses of A and B. That's the
// problem kernel 2 fixes.

#include "kernels.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

bool run_k01_naive(int M, int N, int K, float alpha, const float* A,
                   const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
