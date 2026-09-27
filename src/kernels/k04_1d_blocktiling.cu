// Kernel 4: 1D block tiling (each thread computes several outputs).
//
// In kernel 3 each thread computes ONE output element, so per inner-loop step
// it does 2 shared-memory loads and 1 FMA. Shared memory bandwidth becomes the
// bottleneck. Fix: make each thread compute TM outputs in a column. Now per
// step it loads 1 element of B from smem, and TM elements of A, and does TM
// FMAs. Ratio improves from 2 loads/FMA toward 1 load/FMA.
//
// Typical params from the reference roadmap: BM=64, BN=64, BK=8, TM=8.
// Block has (BM*BN)/TM = 512 threads. Each thread owns an 8x1 strip of C.
//
// Expected gain: 2 to 3x over kernel 3. You should now be at roughly 30 to 40%
// of cuBLAS.
//
// TODO:
//   1. Templated or #define'd BM, BN, BK, TM.
//   2. Cooperative load of a BM x BK tile of A and BK x BN tile of B into smem
//      (each thread loads one element of each; think about the index math).
//   3. float threadResults[TM] = {0};
//   4. for each dotIdx in BK: cache Bs[dotIdx*BN + threadCol] in a register,
//      then for resIdx in TM: threadResults[resIdx] += As[(threadRow*TM+resIdx)*BK + dotIdx] * that register
//   5. Write TM results to C.

#include "kernels.cuh"

__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float* A, const float* B,
                                     float beta, float* C) {
  // TODO
}

bool run_k04_1d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  return false;
}
