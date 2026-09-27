"""Plot results/results.csv.

Produces:
  results/gflops_vs_size.png   line chart, one line per kernel, cuBLAS included
  results/pct_cublas_4096.png  bar chart, % of cuBLAS at the largest size
  results/roofline.png         roofline with each kernel's achieved GFLOPS

Usage: python scripts/plot.py [--peak-tflops 8.1 --bandwidth-gbs 320]
Defaults are the T4. A100: --peak-tflops 19.5 --bandwidth-gbs 1555.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/results.csv")
ap.add_argument("--peak-tflops", type=float, default=8.1)
ap.add_argument("--bandwidth-gbs", type=float, default=320)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
if not rows:
    raise SystemExit("no results yet, run ./sgemm first")

# keep the last measurement for each (kernel, size)
latest = {}
for r in rows:
    latest[(r["kernel"], int(r["size"]))] = r
by_kernel = defaultdict(dict)
for (k, n), r in latest.items():
    by_kernel[k][n] = r
kernels = sorted(by_kernel)
gpu = rows[-1]["gpu"]

# 1. GFLOPS vs size
plt.figure(figsize=(8, 5))
for k in kernels:
    ns = sorted(by_kernel[k])
    plt.plot(ns, [float(by_kernel[k][n]["gflops"]) for n in ns], marker="o", label=k)
plt.xscale("log", base=2)
plt.xlabel("matrix size (N, square)")
plt.ylabel("GFLOPS")
plt.title(f"SGEMM throughput on {gpu}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/gflops_vs_size.png", dpi=150)

# 2. % of cuBLAS at largest size
largest = max(int(r["size"]) for r in rows)
ks = [k for k in kernels if k != "cublas" and largest in by_kernel[k]]
pcts = [float(by_kernel[k][largest]["pct_cublas"]) for k in ks]
plt.figure(figsize=(8, 4.5))
bars = plt.bar(ks, pcts)
for b, p in zip(bars, pcts):
    plt.text(b.get_x() + b.get_width() / 2, p + 1, f"{p:.0f}%", ha="center")
plt.axhline(100, ls="--", color="gray", label="cuBLAS")
plt.ylabel("% of cuBLAS")
plt.title(f"SGEMM at {largest}x{largest} on {gpu}")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig(f"results/pct_cublas_{largest}.png", dpi=150)

# 3. Roofline. SGEMM arithmetic intensity for square N (no reuse assumed):
#    2N^3 flops / (3 * N^2 * 4 bytes) = N/6 flop/byte. Real kernels reuse
#    data, so this is the naive lower bound on intensity; the plot still shows
#    how far each kernel sits from compute peak.
peak = args.peak_tflops * 1e3
bw = args.bandwidth_gbs
ai = np.logspace(-1, 3, 200)
roof = np.minimum(peak, ai * bw)
plt.figure(figsize=(8, 5))
plt.loglog(ai, roof, color="black", label="roofline")
for k in kernels:
    if largest in by_kernel[k]:
        g = float(by_kernel[k][largest]["gflops"])
        plt.scatter([largest / 6], [g], label=k, zorder=3)
plt.xlabel("arithmetic intensity (FLOP/byte, lower bound)")
plt.ylabel("GFLOPS")
plt.title(f"Roofline, {gpu} ({args.peak_tflops} TFLOPS fp32, {args.bandwidth_gbs} GB/s)")
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/roofline.png", dpi=150)
print("wrote results/gflops_vs_size.png, pct_cublas png, roofline.png")
