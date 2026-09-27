// Kernel 2: global memory coalescing.
//
// Same math as kernel 1. The only change is how a thread's ID maps to the
// (row, col) it computes. In kernel 1, adjacent threads in a warp handled
// adjacent ROWS, so their reads of A were strided by K and couldn't be
// combined. Here adjacent threads handle adjacent COLUMNS, so the 32 threads
// of a warp read one shared A element and 32 consecutive B elements.

#include "kernels.cuh"

#define BLOCKSIZE 32

__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
  // 1D block of 1024 threads. threadIdx.x runs 0..1023.
  // Threads 0..31 get row 0 and cols 0..31. Threads 32..63 get row 1, etc.
  const int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  const int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

bool run_k02_coalesced(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  dim3 block(BLOCKSIZE * BLOCKSIZE);  // 1024 threads in a line
  dim3 grid(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
