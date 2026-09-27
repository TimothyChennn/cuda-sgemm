// Kernel 2: global memory coalescing.
//
// The bug in kernel 1: threads in a warp are consecutive in threadIdx.x, and
// threadIdx.x maps to ROW there. So the 32 threads of a warp read 32 different
// rows of A (stride K apart) and the same column of B. Those A reads can't be
// combined into one memory transaction.
//
// The fix: make consecutive threads in a warp handle consecutive COLUMNS of the
// output instead. Then 32 threads read the same A element (broadcast) and 32
// consecutive B elements (one coalesced load). Nothing else changes.
//
// Expected gain on a T4 or A100: roughly 5 to 8x over naive. If you don't see
// that, your row/col mapping is still transposed.
//
// TODO:
//   1. Use a 1D block of BLOCKSIZE*BLOCKSIZE threads.
//   2. row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE
//      col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE
//   3. Same inner loop as kernel 1.
//   4. Set the launcher to return true.

#include "kernels.cuh"

#define BLOCKSIZE 32

__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta,
                                float* C) {
  // TODO
}

bool run_k02_coalesced(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  // dim3 block(BLOCKSIZE * BLOCKSIZE);
  // dim3 grid(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  // sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return false;
}
