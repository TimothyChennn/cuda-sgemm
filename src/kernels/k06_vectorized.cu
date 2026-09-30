// Kernel 6: vectorized loads + transposed A in shared memory.
//
// Same tiling as kernel 5 (128x128 block tile, 8x8 per thread, BK=8).
// Changes:
//   1. Global loads use float4 (128-bit), one per thread per tile for A and B.
//   2. A is stored TRANSPOSED in shared memory (As[k][m] instead of As[m][k]),
//      so each thread's 8 A values for a given k are contiguous and can be
//      read with vector loads instead of 8 strided scalar loads.
//   3. C is written back with float4.

#include "kernels.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C) {
  __shared__ __align__(16) float As[BK * BM];   // transposed: 8 x 128
  __shared__ __align__(16) float Bs[BK * BN];   // 8 x 128

  const int threadCol = threadIdx.x % (BN / TN);  // 0..15
  const int threadRow = threadIdx.x / (BN / TN);  // 0..15

  A += blockIdx.x * BM * K;
  B += blockIdx.y * BN;
  C += blockIdx.x * BM * N + blockIdx.y * BN;

  // A tile is 128 x 8 = 128 rows of 2 float4s = 256 float4s. One per thread.
  const int innerRowA = threadIdx.x / (BK / 4);   // 0..127
  const int innerColA = threadIdx.x % (BK / 4);   // 0..1
  // B tile is 8 x 128 = 8 rows of 32 float4s = 256 float4s. One per thread.
  const int innerRowB = threadIdx.x / (BN / 4);   // 0..7
  const int innerColB = threadIdx.x % (BN / 4);   // 0..31

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    // Load 4 floats of A in one instruction, then scatter them transposed.
    float4 tmp = reinterpret_cast<const float4*>(
        &A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // Load 4 floats of B in one instruction, store them as-is.
    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Both of these are now 8 contiguous floats: compiler emits vector loads.
      for (int i = 0; i < TM; ++i)
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      for (int i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
          threadResults[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }

  // Write back 4 outputs per instruction.
  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; n += 4) {
      float4* dst = reinterpret_cast<float4*>(
          &C[(threadRow * TM + m) * N + threadCol * TN + n]);
      float4 c = dst[0];
      c.x = alpha * threadResults[m * TN + n + 0] + beta * c.x;
      c.y = alpha * threadResults[m * TN + n + 1] + beta * c.y;
      c.z = alpha * threadResults[m * TN + n + 2] + beta * c.z;
      c.w = alpha * threadResults[m * TN + n + 3] + beta * c.w;
      dst[0] = c;
    }
  }
}

bool run_k06_vectorized(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
  dim3 block((BM * BN) / (TM * TN));   // 256 threads
  dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
  sgemm_vectorized<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
