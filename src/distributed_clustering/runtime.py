from __future__ import annotations

import ctypes as ct
import csv
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

BACKENDS = ("cpu", "openmp", "torch-cpu", "mps", "cuda-naive", "cuda-optimized", "cublas")
FLOAT = ct.POINTER(ct.c_float)
INT = ct.POINTER(ct.c_int)
DOUBLE = ct.POINTER(ct.c_double)
BYTE = ct.POINTER(ct.c_ubyte)


def pointer(array, dtype=FLOAT):
    return array.ctypes.data_as(dtype)


def matrix(value, name="embeddings"):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    if a.size > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds the supported int32 element count")
    return np.ascontiguousarray(a)


def normalize(value):
    a = matrix(value)
    norms = np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
    return np.ascontiguousarray(a / np.where(norms > 0, norms, 1.0), dtype=np.float32)


def load_matrix(path):
    return normalize(np.load(path, allow_pickle=False))


class Native:
    def __init__(self):
        build = Path(os.environ.get("DISTRIBUTED_CLUSTERING_BUILD_DIR", str(ROOT / "build")))
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        self.cpu = None
        self.cuda = None
        self.cpu_error = "Run python scripts/build.py first"
        self.cuda_error = "CUDA library was not built; NVIDIA hardware and nvcc are required"
        try:
            self.cpu = ct.CDLL(str(build / ("libdistributed_clustering_ops" + suffix)))
            self.cpu.hpc_error.restype = ct.c_char_p
            self.cpu.hpc_has_openmp.restype = ct.c_int
            self.cpu.hpc_kmeans.argtypes = [FLOAT, FLOAT, INT] + [ct.c_int] * 6 + [DOUBLE]
        except OSError as exc:
            self.cpu_error = str(exc)
        cuda_path = build / ("libdistributed_clustering_cuda" + suffix)
        if not cuda_path.exists():
            return
        try:
            self.cuda = ct.CDLL(str(cuda_path))
            self.cuda.hpc_cuda_error.restype = ct.c_char_p
            self.cuda.hpc_cuda_devices.restype = ct.c_int
            self.cuda.hpc_cuda_kmeans.argtypes = [FLOAT, FLOAT, INT] + [ct.c_int] * 5 + [DOUBLE]
        except OSError as exc:
            self.cuda_error = str(exc)

    def check(self, status, cuda=False):
        if status:
            library = self.cuda if cuda else self.cpu
            message = library.hpc_cuda_error() if cuda else library.hpc_error()
            raise RuntimeError(message.decode())


_native = None


def native():
    global _native
    if _native is None:
        _native = Native()
    return _native


def availability(backend):
    if backend not in BACKENDS:
        return False, "Unknown backend"
    lib = native()
    if backend in ("cpu", "openmp"):
        if lib.cpu is None:
            return False, lib.cpu_error
        if backend == "openmp" and not lib.cpu.hpc_has_openmp():
            return False, "OpenMP was not compiled in"
    elif backend in ("torch-cpu", "mps"):
        try:
            import torch
        except ImportError as exc:
            return False, str(exc)
        if backend == "mps" and not torch.backends.mps.is_available():
            return False, "MPS is unavailable in this process; try a regular macOS terminal"
    elif lib.cuda is None:
        return False, lib.cuda_error
    elif not lib.cuda.hpc_cuda_devices():
        return False, lib.cuda.hpc_cuda_error().decode()
    return True, "available"


def require(backend):
    ok, reason = availability(backend)
    if not ok:
        raise RuntimeError(f"{backend}: {reason}")


def cuda_mode(backend):
    return {"cuda-naive": 0, "cuda-optimized": 1, "cublas": 2}[backend]


def synchronize(backend):
    if backend == "mps":
        import torch
        torch.mps.synchronize()


def benchmark(operation, backend, repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    operation()  # Untimed warm-up, including lazy library initialization.
    synchronize(backend)
    samples, result = [], None
    for _ in range(repeats):
        synchronize(backend)
        start = time.perf_counter()
        result = operation()
        synchronize(backend)
        samples.append((time.perf_counter() - start) * 1000)
    return result, {"median_ms": float(np.median(samples)), "min_ms": min(samples),
                    "samples_ms": samples, "repeats": repeats, "warmup_runs": 1}


def metadata():
    import importlib.metadata
    versions = {}
    for name in ('numpy', 'torch', 'matplotlib', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0], "packages": versions,
            "backends": {b: {"available": availability(b)[0], "reason": availability(b)[1]} for b in BACKENDS}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_common_arguments(parser, default_output):
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / default_output)


def validate_common(args):
    if args.threads < 1 or args.repeats < 1:
        raise ValueError("threads and repeats must be positive")
    import torch
    torch.set_num_threads(args.threads)


def timing_plot(path, records, title, field="median_ms"):
    import matplotlib.pyplot as plt
    usable = [r for r in records if r.get("status") == "ok" and field in r]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([r["backend"] for r in usable], [r[field] for r in usable], color="#287a9c")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
