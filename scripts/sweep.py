#!/usr/bin/env python3
"""Measure distributed-clustering problem-size and thread scaling."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from distributed_clustering.algorithms import initial_centers, kmeans
from distributed_clustering.data import cluster_demo
from distributed_clustering.runtime import BACKENDS, availability, benchmark, metadata, output_directory, write_csv, write_json
from distributed_clustering.app import quality

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 8000, 32000])
parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["cpu", "openmp"])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--output", type=Path, default=ROOT / "results/scaling")
args = parser.parse_args()
if args.dim < 2 or args.repeats < 1 or min(args.threads) < 1 or min(args.sizes) < 12:
    parser.error("dim>=2, positive repeats/threads, and sizes>=12 are required")
out = output_directory(args.output)
records = []
for size in args.sizes:
    x, truth, anomalous = cluster_demo(size, args.dim, 12, 0.03, 42)
    initial = initial_centers(x, 12, 42)
    operation = lambda b, t: kmeans(x, initial, 10, b, t)
    reference = operation("cpu", 1)
    for backend in dict.fromkeys(args.backends):
        ok, reason = availability(backend)
        if not ok:
            records.append({"size": size, "backend": backend, "status": "skipped", "reason": reason})
            continue
        for threads in ([1] if backend == "cpu" else args.threads):
            import torch
            torch.set_num_threads(threads)
            result, elapsed = benchmark(lambda: operation(backend, threads), backend, args.repeats)
            quality(x, result[0], result[1], result[2], truth, anomalous)
            records.append({"size": size, "backend": backend, "threads": threads, "status": "ok", **elapsed})
            print(f"size={size} {backend} threads={threads}: {elapsed['median_ms']:.3f} ms", flush=True)
write_json(out / "results.json", {"project": "distributed-clustering", "environment": metadata(), "dim": args.dim,
           "timing_scope": "Warm API wall time after one warm-up. 12 clusters, 10 updates; initialization excluded; validation/allocation and transfers included.", "records": records})
write_csv(out / "timings.csv", records)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
series = sorted({(row["backend"], row["threads"]) for row in records if row["status"] == "ok"})
for backend, threads in series:
    rows = sorted([row for row in records if row["status"] == "ok" and row["backend"] == backend and row["threads"] == threads], key=lambda row: row["size"])
    ax.plot([row["size"] for row in rows], [row["median_ms"] for row in rows], "o-", label=f"{backend}, {threads} thread(s)")
ax.set(xlabel="Rows", ylabel="Wall-clock time (ms)", title="distributed-clustering: size and thread scaling")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(out / "scaling.png", dpi=160)
plt.close(fig)
print("Results:", out)
