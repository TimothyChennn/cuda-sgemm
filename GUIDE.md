# Working guide

## Order of operations

1. Read `src/kernels/k01_naive.cu` until you can explain every line.
2. Read `src/runner.cu` once so you know what "correct" and "fast" mean here.
3. Do kernels 2 through 6 in order. Each file has the idea, the expected
   gain, and a TODO list. Do not skip ahead; each one motivates the next.
4. After each kernel: run it at all sizes, confirm "OK", note the % of
   cuBLAS in README, and write two sentences on why it helped.
5. After kernel 6: plot, write the README, push.

## Reading, in order (do this alongside, not before)

- Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like
  Performance" (siboehm.com). This repo follows his kernel sequence. Read the
  section for kernel N before writing kernel N, then close it and write.
- Programming Massively Parallel Processors (PMPP), chapters 1 to 6. Chapter 5
  (memory) and 6 (performance) are the ones interviewers pull from.
- NVIDIA CUDA C++ Programming Guide, "Performance Guidelines" section.

## Concepts to be able to explain out loud (this is the interview)

- thread, warp (32 threads), block, grid, SM. How blocks are scheduled to SMs.
- Why a warp reading 32 consecutive floats is one transaction and 32 strided
  floats is 32 transactions (coalescing).
- Global vs shared vs register memory: latency, size, scope.
- Bank conflicts in shared memory and how a transposed layout can cause or
  avoid them.
- Occupancy: what limits it (registers, smem, block size) and why max
  occupancy isn't always fastest.
- Arithmetic intensity and the roofline. Where does SGEMM sit and why can it
  reach compute peak.
- Why cuBLAS is still faster than kernel 6 (tensor cores, warp tiling,
  double buffering, tuned tile sizes per architecture).

## Ground rules

- You write the kernels. Use Claude Code or Claude to unblock you when
  you're stuck for more than 30 minutes, but ask for the concept, not the
  code. If you paste in a kernel you can't explain, this project is worth
  nothing in an interview.
- Every kernel gets a number in the README before you move on.
- Commit after every kernel.

## Rough timeline

- Days 1 to 3: LeetGPU easy problems, kernel 1 read-through, kernel 2.
- Days 4 to 7: kernel 3. Profile it. Understand why the gain is smaller
  than expected.
- Days 8 to 12: kernel 4 and 5. Kernel 5 is the hard one; budget for it.
- Days 13 to 16: kernel 6, ptxas checks, A100 run.
- Days 17 to 21: plots, README, short write-up. Push. Post it.

## Stretch (after the 3 weeks, over winter)

- Kernel 7: warp tiling. Gets you to 90%+ of cuBLAS on CUDA cores.
- Kernel 8: tensor cores via wmma or mma.sync. Different game, fp16 inputs.
- Then flash attention in Triton, which is the second named artifact.
