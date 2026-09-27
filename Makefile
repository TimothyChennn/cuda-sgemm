# Build:  make            (defaults to sm_75, the Colab T4)
#         make ARCH=sm_80  (A100)
#         make ARCH=sm_89  (L4 / RTX 40xx)
# Run:    ./sgemm <kernel_id> [sizes...]
# Reg usage per kernel: make ptxas

ARCH ?= sm_75
NVCC ?= nvcc
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++17 -Isrc
LDFLAGS = -lcublas

SRCS = src/runner.cu $(wildcard src/kernels/*.cu)

sgemm: $(SRCS) src/kernels/kernels.cuh
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(LDFLAGS)

ptxas: $(SRCS)
	$(NVCC) $(NVCCFLAGS) --ptxas-options=-v $(SRCS) -o /dev/null $(LDFLAGS)

# Run every kernel at every default size and log to results/results.csv
bench: sgemm
	@for k in 0 1 2 3 4 5 6; do ./sgemm $$k; done

clean:
	rm -f sgemm

.PHONY: ptxas bench clean
