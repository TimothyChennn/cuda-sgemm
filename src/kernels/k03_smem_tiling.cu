// Kernel 3: shared memory tiling.
//
// Kernel 2 still reads every A and B element from global memory K times over
// across the block. Global memory is slow (hundreds of cycles). Shared memory
// is on-chip and fast (a few cycles) but small (48 KB default per block).
//
// Idea: each block computes a BLOCKSIZE x BLOCKSIZE tile of C. Loop over K in
// chunks of BLOCKSIZE. For each chunk, cooperatively load a BLOCKSIZE x BLOCKSIZE
// tile of A and of B into shared memory, __syncthreads(), do the partial dot
// products from shared memory, __syncthreads(), advance.
//
// Every global element is now loaded once per block instead of once per thread.
//
// Expected gain: maybe 1.5 to 2x over kernel 2. Less than you'd hope, which is
// the interesting part. Profile it and figure out why (hint: arithmetic
// intensity per thread is still low, each thread does one FMA per two smem
// loads). That's what kernel 4 fixes.
//
// TODO:
//   1. __shared__ float As[BLOCKSIZE * BLOCKSIZE]; same for Bs.
//   2. Advance A, B, C pointers to this block's starting tile.
//   3. for (bk = 0; bk < K; bk += BLOCKSIZE):
//        load As[threadRow*BLOCKSIZE + threadCol] = A[threadRow*K + threadCol]
//        load Bs[...] = B[threadRow*N + threadCol]
//        __syncthreads()
//        A += BLOCKSIZE; B += BLOCKSIZE * N;
//        for (i = 0; i < BLOCKSIZE; i++) acc += As[threadRow*BLOCKSIZE+i] * Bs[i*BLOCKSIZE+threadCol]
//        __syncthreads()
//   4. Write C.
//   Assumes M, N, K are multiples of BLOCKSIZE. All benchmark sizes are.

#include "kernels.cuh"

#define BLOCKSIZE 32

__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C) {
  // TODO
}

bool run_k03_smem_tiling(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  return false;
}
