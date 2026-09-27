// Kernel 6: vectorized global loads and transposed A in shared memory.
//
// Two changes on top of kernel 5:
//   a) Load from global memory as float4 (128-bit loads) instead of float.
//      Fewer, wider memory instructions. Requires 16-byte alignment, which
//      holds for our sizes.
//   b) Store the A tile TRANSPOSED in shared memory so that the inner loop
//      reads As as contiguous float4s too, and the loads from smem into
//      regM become vectorized. Watch for bank conflicts.
//
// Also worth trying here: __launch_bounds__ to control occupancy, and
// #pragma unroll on the inner loops.
//
// Expected: 75 to 85% of cuBLAS on fp32 CUDA cores. Beyond this, the next
// step is warp tiling and then tensor cores (wmma / mma.sync), which is the
// "stretch" phase in the README.
//
// TODO:
//   1. Copy kernel 5.
//   2. Replace the global->smem loads with reinterpret_cast<const float4*>.
//   3. Write As transposed: As[(innerColA*4+i)*BM + innerRowA] = tmp.{x,y,z,w}
//   4. Update the regM loads to read As[dotIdx*BM + threadRow*TM + i].
//   5. Vectorize the C store as float4 too.

#include "kernels.cuh"

__global__ void sgemm_vectorized(int M, int N, int K, float alpha,
                                 const float* A, const float* B, float beta,
                                 float* C) {
  // TODO
}

bool run_k06_vectorized(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C) {
  return false;
}
