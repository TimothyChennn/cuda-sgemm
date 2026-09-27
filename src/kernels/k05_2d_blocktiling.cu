// Kernel 5: 2D block tiling (each thread computes a TM x TN tile).
//
// Kernel 4 improved the load/FMA ratio in one dimension. Do it in both. Each
// thread now owns a TM x TN patch of C. Per inner step: load TM values of A
// and TN values of B from smem into registers, then do TM*TN FMAs from
// registers. Ratio is now (TM+TN) loads per TM*TN FMAs. With TM=TN=8 that's
// 16 loads per 64 FMAs, four times better than kernel 4.
//
// Typical params: BM=128, BN=128, BK=8, TM=8, TN=8. 256 threads per block.
// Watch register pressure: 64 accumulators + 16 cache registers per thread.
// If nvcc spills to local memory, performance dies. Check with
//   nvcc --ptxas-options=-v
//
// Expected: 60 to 70% of cuBLAS. This is the kernel where it starts to look
// like a real GEMM.
//
// TODO:
//   1. float threadResults[TM*TN], regM[TM], regN[TN].
//   2. Cooperative smem load: each thread now loads multiple elements of A and
//      B per tile (BM*BK / numThreads of A, BK*BN / numThreads of B). Use a
//      strided loop.
//   3. Triple loop: for dotIdx in BK { load regM, regN; for i in TM for j in TN
//      threadResults[i*TN+j] += regM[i]*regN[j]; }
//   4. Write the TM x TN patch to C.

#include "kernels.cuh"

__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                     const float* A, const float* B,
                                     float beta, float* C) {
  // TODO
}

bool run_k05_2d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  return false;
}
