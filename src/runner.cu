// SGEMM benchmark harness.
// Computes C = alpha * A @ B + beta * C, all row-major, M x K times K x N.
//
// Usage:  ./sgemm <kernel_id> [size ...]
//   kernel_id 0 = cuBLAS reference
//   kernel_id 1..6 = your kernels (see src/kernels/)
//   sizes default to 256 512 1024 2048 4096 (square matrices)
//
// For each size: verifies your result against cuBLAS, times it with CUDA events,
// prints GFLOPS and % of cuBLAS, and appends a row to results/results.csv.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kernels/kernels.cuh"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),      \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t st = (call);                                                 \
    if (st != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st, __FILE__,          \
              __LINE__);                                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

static void fill_random(std::vector<float>& v, unsigned seed) {
  srand(seed);
  for (auto& x : v) x = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

// cuBLAS is column-major. To get row-major C = A@B we compute C^T = B^T A^T,
// which in column-major terms is just swapping the operand order.
static void run_cublas(cublasHandle_t h, int M, int N, int K, float alpha,
                       const float* A, const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
}

static bool run_kernel(int id, cublasHandle_t h, int M, int N, int K,
                       float alpha, const float* A, const float* B, float beta,
                       float* C) {
  switch (id) {
    case 0: run_cublas(h, M, N, K, alpha, A, B, beta, C); return true;
    case 1: return run_k01_naive(M, N, K, alpha, A, B, beta, C);
    case 2: return run_k02_coalesced(M, N, K, alpha, A, B, beta, C);
    case 3: return run_k03_smem_tiling(M, N, K, alpha, A, B, beta, C);
    case 4: return run_k04_1d_blocktiling(M, N, K, alpha, A, B, beta, C);
    case 5: return run_k05_2d_blocktiling(M, N, K, alpha, A, B, beta, C);
    case 6: return run_k06_vectorized(M, N, K, alpha, A, B, beta, C);
    default:
      fprintf(stderr, "unknown kernel id %d\n", id);
      exit(1);
  }
}

static const char* kernel_name(int id) {
  static const char* names[] = {"cublas",         "01_naive",
                                "02_coalesced",   "03_smem_tiling",
                                "04_1d_blocktile", "05_2d_blocktile",
                                "06_vectorized"};
  return names[id];
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <kernel_id 0-6> [size ...]\n", argv[0]);
    return 1;
  }
  int id = atoi(argv[1]);
  std::vector<int> sizes;
  for (int i = 2; i < argc; i++) sizes.push_back(atoi(argv[i]));
  if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s | kernel: %s\n", prop.name, kernel_name(id));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  FILE* csv = fopen("results/results.csv", "a");
  if (!csv) {
    fprintf(stderr, "could not open results/results.csv (run from repo root)\n");
    return 1;
  }

  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 3, iters = 10;

  for (int n : sizes) {
    int M = n, N = n, K = n;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    std::vector<float> hC((size_t)M * N), hRef((size_t)M * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dRef, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    CUDA_CHECK(cudaMemset(dRef, 0, bytesC));

    // ---- correctness check against cuBLAS ----
    run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef);
    if (!run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC)) {
      printf("  size %5d: kernel %s not implemented yet, skipping\n", n,
             kernel_name(id));
      cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
      continue;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytesC, cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (size_t i = 0; i < hC.size(); i++)
      max_err = fmax(max_err, fabs((double)hC[i] - hRef[i]));
    bool ok = max_err < 1e-2 * K / 256.0;  // loose tolerance, scales with K

    // ---- timing: your kernel ----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < warmup; i++)
      run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
      run_kernel(id, handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    // ---- timing: cuBLAS on the same problem ----
    for (int i = 0; i < warmup; i++)
      run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef);
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
      run_cublas(handle, M, N, K, alpha, dA, dB, beta, dRef);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_cublas = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, t0, t1));
    ms_cublas /= iters;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e-3) / 1e9;
    double gflops_cublas = flops / (ms_cublas * 1e-3) / 1e9;
    double pct = 100.0 * gflops / gflops_cublas;

    printf("  size %5d: %8.3f ms  %8.1f GFLOPS  %6.1f%% of cuBLAS  max_err=%.2e %s\n",
           n, ms, gflops, pct, max_err, ok ? "OK" : "WRONG RESULT");
    fprintf(csv, "%s,%s,%d,%.4f,%.1f,%.1f,%.1f,%s\n", prop.name,
            kernel_name(id), n, ms, gflops, gflops_cublas, pct,
            ok ? "ok" : "wrong");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
  }

  fclose(csv);
  cublasDestroy(handle);
  return 0;
}
