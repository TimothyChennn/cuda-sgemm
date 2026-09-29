// Kernel 3: shared memory tiling.
//
// Each block computes a 32x32 tile of C. Instead of every thread reading its
// own row of A and column of B from global memory, the block cooperatively
// loads a 32x32 tile of A and of B into shared memory (fast, on-chip), all
// threads compute from that, then the tiles slide along K.

#include "kernels.cuh"

#define BLOCKSIZE 32

__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  // Fast on-chip scratchpad, shared by all 1024 threads in this block.
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // My position inside the 32x32 block.
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  // Move the pointers to the top-left corner of this block's tile.
  A += blockIdx.x * BLOCKSIZE * K;
  B += blockIdx.y * BLOCKSIZE;
  C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE;

  float acc = 0.0f;

  for (int bk = 0; bk < K; bk += BLOCKSIZE) {
    // Every thread loads exactly one element of each tile.
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // Wait for all 1024 loads to finish before anyone reads the tile.
    __syncthreads();

    // Slide to the next tile along K.
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // Dot product across the tile, from fast memory.
    for (int i = 0; i < BLOCKSIZE; ++i) {
      acc += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
    }

    // Wait until everyone is done reading before the next load overwrites it.
    __syncthreads();
  }

  C[threadRow * N + threadCol] = alpha * acc + beta * C[threadRow * N + threadCol];
}

bool run_k03_smem_tiling(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  sgemm_smem<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
