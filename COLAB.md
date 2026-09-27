# Running this on Google Colab (free T4)

You have a MacBook, so all GPU work runs in Colab. Each cell below is one
Colab cell. Runtime > Change runtime type > T4 GPU first.

## Cell 1: check the GPU
```
!nvidia-smi
!nvcc --version
```
You want to see "Tesla T4" and CUDA 12.x.

## Cell 2: get the repo
Push this folder to GitHub first (see README), then:
```
!git clone https://github.com/TimothyChennn/cuda-sgemm.git
%cd cuda-sgemm
```
If you'd rather not push yet, upload the zip via the Files panel and:
```
!unzip -q cuda-sgemm.zip && cd cuda-sgemm
```

## Cell 3: build
```
!make ARCH=sm_75
```
Zero warnings expected. If a kernel file fails to compile, fix it and rerun
just this cell.

## Cell 4: run one kernel
```
!./sgemm 1
```
Kernel 0 is cuBLAS. Kernels you haven't implemented print "not implemented".

## Cell 5: run everything and plot
```
!make bench
!pip -q install matplotlib
!python scripts/plot.py
```
Then open results/*.png in the Files panel, or:
```
from IPython.display import Image, display
display(Image("results/gflops_vs_size.png"))
```

## Cell 6: register and spill check (kernels 5 and 6)
```
!make ptxas 2>&1 | grep -E "Function|registers|spill"
```
If you see "bytes spill stores" above 0 on kernel 5 or 6, you have register
pressure. Reduce TM/TN or restructure.

## Cell 7: profile a kernel (optional but this is the real skill)
```
!ncu --set full --kernel-name regex:sgemm_smem ./sgemm 3 4096 > results/ncu_k03.txt 2>&1
!head -80 results/ncu_k03.txt
```
Look at: Memory Throughput %, Compute (SM) Throughput %, Achieved Occupancy,
and the "Memory Workload Analysis" section. This is where you learn WHY a
kernel is slow, not just that it is. Colab may not have ncu installed on
free tier; if not, skip until you have a paid runtime or a rented GPU.

## Saving results between sessions
Colab wipes the VM. Commit results/results.csv and the pngs to GitHub at the
end of each session:
```
!git config user.email "tic058@ucsd.edu" && git config user.name "Timothy Chen"
!git add results && git commit -m "results: kernel N on T4" && git push
```
(You'll need a GitHub token for push from Colab; easiest is to download the
csv and commit from your Mac.)

## When you're done on the T4
Rerun the full bench on an A100 (Colab Pro, ~$10) for the final README numbers.
`make ARCH=sm_80` and `python scripts/plot.py --peak-tflops 19.5 --bandwidth-gbs 1555`.
