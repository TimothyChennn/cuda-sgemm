// Kernel 4: 1D block tiling. Each thread computes TM=8 outputs in a column.
//
// Block tile: BM x BN = 64x64 outputs. K is walked in chunks of BK=8.
// 512 threads per block, each owning an 8x1 strip of C.
// Per inner step a thread loads 1 value of B into a register and reuses it
// for 8 FMAs, so the load:FMA ratio drops from 2:1 (kernel 3) to about 9:8.

#include "kernels.cuh"

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float* A, const float* B,
                                     float beta, float* C) {
  __shared__ float As[BM * BK];   // 64 rows x 8 cols of A
  __shared__ float Bs[BK * BN];   // 8 rows x 64 cols of B

  // Which 64x64 tile of C this block owns.
  A += blockIdx.x * BM * K;
  B += blockIdx.y * BN;
  C += blockIdx.x * BM * N + blockIdx.y * BN;

  // My column within the tile (0..63), and which 8-row strip I own (0..7).
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // Which element I load when we fill the tiles. A's tile is 64x8 = 512
  // elements, B's is 8x64 = 512, one per thread. The A mapping puts
  // consecutive threads on consecutive columns so the global load coalesces.
  const int innerColA = threadIdx.x % BK;
  const int innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN;
  const int innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0f};   // 8 accumulators in registers

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      const float tmpB = Bs[dotIdx * BN + threadCol];   // load B once...
      for (int resIdx = 0; resIdx < TM; ++resIdx) {     // ...reuse it 8 times
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    const int r = threadRow * TM + resIdx;
    C[r * N + threadCol] = alpha * threadResults[resIdx] + beta * C[r * N + threadCol];
  }
}

bool run_k04_1d_blocktiling(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  dim3 block((BM * BN) / TM);   // 512 threads
  dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
  sgemm_1d_blocktiling<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
