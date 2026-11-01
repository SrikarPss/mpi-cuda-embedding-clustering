from __future__ import annotations

import numpy as np
from .runtime import BYTE, DOUBLE, INT, cuda_mode, matrix, native, normalize, pointer, require


def initial_centers(x, clusters, seed=42):
    """Seeded cosine-distance sampling, independent of implementation backend."""
    x = matrix(x)
    if not 1 <= clusters <= len(x):
        raise ValueError("1 <= clusters <= number of points is required")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(x)))]
    distances = np.maximum(0, 1 - x @ x[chosen[0]])
    while len(chosen) < clusters:
        weights = distances.astype(np.float64)
        weights[chosen] = 0
        total = weights.sum()
        if total <= 1e-12:
            remaining = np.setdiff1d(np.arange(len(x)), chosen)
            item = int(rng.choice(remaining))
        else:
            item = int(rng.choice(len(x), p=weights / total))
        chosen.append(item)
        distances = np.minimum(distances, np.maximum(0, 1 - x @ x[item]))
    return x[chosen].copy()


def kmeans(x, centers, iterations=15, backend="cpu", threads=4):
    require(backend)
    x, centers = matrix(x), matrix(centers).copy()
    n, dim = x.shape
    clusters = len(centers)
    if centers.shape[1] != dim or not 1 <= clusters <= n or iterations < 1 or threads < 1:
        raise ValueError("Invalid k-means dimensions, clusters, iterations, or threads")
    labels = np.empty(n, np.int32)
    objectives = np.empty(iterations + 1, np.float64)
    lib = native()
    if backend in ("cpu", "openmp"):
        status = lib.cpu.hpc_kmeans(pointer(x), pointer(centers), pointer(labels, INT), n, dim, clusters,
                                    iterations, threads, int(backend == "openmp"), pointer(objectives, DOUBLE))
        lib.check(status)
    elif backend.startswith("cuda") or backend == "cublas":
        status = lib.cuda.hpc_cuda_kmeans(pointer(x), pointer(centers), pointer(labels, INT), n, dim, clusters,
                                          iterations, cuda_mode(backend), pointer(objectives, DOUBLE))
        lib.check(status, cuda=True)
    else:
        import torch
        device = "mps" if backend == "mps" else "cpu"
        with torch.inference_mode():
            tx, tc = torch.as_tensor(x, device=device), torch.as_tensor(centers, device=device)
            for it in range(iterations + 1):
                parts, objective = [], 0.0
                for start in range(0, n, 4096):
                    best, assigned = torch.max(tx[start:start + 4096] @ tc.T, dim=1)
                    objective += float((1 - best).sum().item())
                    parts.append(assigned)
                assigned = torch.cat(parts)
                objectives[it] = objective
                if it == iterations:
                    break
                sums = torch.zeros((clusters, dim), device=device)
                sums.index_add_(0, assigned, tx)
                norms = torch.linalg.vector_norm(sums, dim=1, keepdim=True)
                tc = torch.where(norms > 1e-15, sums / torch.clamp(norms, min=1e-15), tc)
            centers[:] = tc.cpu().numpy()
            labels[:] = assigned.cpu().numpy()
    return centers, labels, objectives
