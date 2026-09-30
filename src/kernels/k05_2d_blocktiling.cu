// Kernel 5: 2D block tiling. Each thread computes a TM x TN = 8x8 patch of C.
//
// Block tile: BM x BN = 128x128 outputs, K walked in chunks of BK=8.
// 256 threads per block. Per inner step, a thread loads 8 values of A and
// 8 of B into registers and does 64 FMAs from them: 16 loads per 64 FMAs.

#include "kernels.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_2d_blocktiling(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  __shared__ float As[BM * BK];   // 128 x 8
  __shared__ float Bs[BK * BN];   // 8 x 128

  const int numThreads = (BM * BN) / (TM * TN);   // 256

  // Which 8x8 patch of the 128x128 tile I own. 16x16 grid of patches.
  const int threadCol = threadIdx.x % (BN / TN);  // 0..15
  const int threadRow = threadIdx.x / (BN / TN);  // 0..15

  A += blockIdx.x * BM * K;
  B += blockIdx.y * BN;
  C += blockIdx.x * BM * N + blockIdx.y * BN;

  // Load mapping. A tile has 1024 elements, 256 threads, so each thread
  // loads 4 of them, strideA rows apart. Same for B.
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int strideA = numThreads / BK;            // 32
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;
  const int strideB = numThreads / BN;            // 2

  float threadResults[TM * TN] = {0.0f};          // 64 accumulators
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    for (int off = 0; off < BM; off += strideA) {
      As[(innerRowA + off) * BK + innerColA] =
          A[(innerRowA + off) * K + innerColA];
    }
    for (int off = 0; off < BK; off += strideB) {
      Bs[(innerRowB + off) * BN + innerColB] =
          B[(innerRowB + off) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Pull my 8 A values and 8 B values into registers.
      for (int i = 0; i < TM; ++i)
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      for (int i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      // 64 FMAs, all from registers.
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          threadResults[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }

  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      const int r = threadRow * TM + m;
      const int c = threadCol * TN + n;
      C[r * N + c] = alpha * threadResults[m * TN + n] + beta * C[r * N + c];
    }
  }
}

bool run_k05_2d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  dim3 block((BM * BN) / (TM * TN));   // 256 threads
  dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
  sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
