import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from .algorithms import initial_centers, kmeans
from .data import cluster_demo
from .runtime import (ROOT, add_common_arguments, availability, benchmark, load_matrix, metadata,
                      output_directory, timing_plot, validate_common, write_csv, write_json)


def quality(x, centers, labels, objectives, truth=None, anomalous=None):
    if not np.isfinite(centers).all() or not np.isfinite(objectives).all():
        raise RuntimeError("Nonfinite clustering output")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise RuntimeError("Invalid cluster labels")
    if np.any(np.diff(objectives) > max(1, abs(objectives[0]))*2e-5):
        raise RuntimeError("Spherical k-means objective increased beyond roundoff tolerance")
    scores = np.einsum("ij,ij->i", x, centers[labels])
    expected_objective = float(np.sum(1-scores, dtype=np.float64))
    if not np.isclose(objectives[-1], expected_objective, rtol=2e-5, atol=1e-4):
        raise RuntimeError("Reported objective does not match final labels and centers")
    for start in range(0, len(x), 2048):
        best = np.max(x[start:start+2048] @ centers.T, axis=1)
        if np.any(best-scores[start:start+2048] > 3e-5):
            raise RuntimeError("Final labels are not nearest-center assignments")
    norms = np.linalg.norm(centers, axis=1)
    if not np.all((norms < 1e-10) | (np.abs(norms-1) < 2e-4)):
        raise RuntimeError("Nonzero centers must be unit normalized")
    distances = 1-scores
    ranking = np.lexsort((np.arange(len(x)), -distances))
    result = {"objective": float(objectives[-1]), "mean_cosine_distance": expected_objective/len(x),
              "empty_clusters": int(np.sum(np.bincount(labels, minlength=len(centers)) == 0))}
    if truth is not None:
        from scipy.optimize import linear_sum_assignment
        valid = truth >= 0
        if valid.any():
            confusion = np.zeros((int(truth[valid].max())+1, len(centers)), dtype=np.int64)
            np.add.at(confusion, (truth[valid], labels[valid]), 1)
            rows, columns = linear_sum_assignment(-confusion)
            result["inlier_label_accuracy_after_matching"] = float(confusion[rows, columns].sum()/valid.sum())
    if anomalous is not None and anomalous.any():
        count = int(anomalous.sum())
        result["injected_outlier_recall_at_outlier_count"] = float(anomalous[ranking[:count]].mean())
    return result, distances, ranking


def reports(out, backend, centers, labels, x, distances, ranking):
    rows = []
    for c in range(len(centers)):
        members = np.flatnonzero(labels == c)
        representative = -1
        score = None
        if len(members):
            local = x[members] @ centers[c]
            representative = int(members[np.argmax(local)])
            score = float(local.max())
        rows.append({"cluster": c, "size": len(members), "representative_id": representative,
                     "representative_cosine": score})
    write_csv(out / f"{backend}-clusters.csv", rows)
    write_csv(out / f"{backend}-outliers.csv", [{"rank": rank+1, "point_id": int(i), "cluster": int(labels[i]),
                                                "cosine_distance": float(distances[i])}
                                               for rank, i in enumerate(ranking[:min(200, len(x))])])


def plot_objectives(path, histories):
    if not histories:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, values in histories.items():
        ax.plot(np.arange(len(values)), values, label=backend)
    ax.set(xlabel="Completed centroid updates", ylabel="Sum of cosine distances", title="Spherical k-means convergence")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_mpi(out, args, x, backend, ranks):
    build = Path(os.environ.get("DISTRIBUTED_CLUSTERING_BUILD_DIR", str(ROOT / "build")))
    executable = build / "hpc_cluster_mpi"
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    launcher = shutil.which("mpirun", path=env["PATH"])
    if not executable.exists() or not launcher:
        return {"backend": f"mpi-{backend}-{ranks}", "status": "skipped", "reason": "MPI executable or mpirun unavailable"}, None
    prefix = out / f"mpi-{backend}-{ranks}"
    command = [launcher, "--oversubscribe", "-n", str(ranks)]
    if args.mpi_hosts:
        command += ["--host", args.mpi_hosts]
    command += [str(executable), "--input", str(out / "points.f32"), "--initial", str(out / "initial.f32"),
                "--rows", str(len(x)), "--dim", str(x.shape[1]), "--clusters", str(args.clusters),
                "--iterations", str(args.iterations), "--threads", str(args.threads),
                "--backend", backend, "--output", str(prefix)]
    samples = []
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        try:
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.mpi_timeout)
            Path(str(prefix) + ".log").write_text(completed.stdout + completed.stderr)
        except subprocess.TimeoutExpired:
            return {"backend": prefix.name, "status": "error", "reason": "MPI launch timed out", "command": command}, None
        if completed.returncode:
            message = completed.stdout + completed.stderr
            denied = "operation not permitted" in message.lower() or "permission denied" in message.lower()
            return {"backend": prefix.name, "status": "blocked" if denied else "error",
                    "reason": "MPI launch failed; see " + str(prefix) + ".log", "command": command}, None
        if repeat:
            samples.append((time.perf_counter()-start)*1000)
    centers = np.fromfile(str(prefix) + ".centers.f32", dtype=np.float32).reshape(args.clusters, x.shape[1])
    labels = np.fromfile(str(prefix) + ".labels.i32", dtype=np.int32)
    details = json.loads(Path(str(prefix) + ".json").read_text())
    record = {"backend": prefix.name, "status": "ok", "median_ms": float(np.median(samples)),
              "min_ms": min(samples), "samples_ms": samples, "ranks": ranks, "warmup_runs": 1,
              "compute_loop_ms": details["compute_loop_seconds"]*1000,
              "max_rank_compute_ms": details["max_rank_compute_seconds"]*1000,
              "max_rank_collective_ms": details["max_rank_collective_seconds"]*1000}
    return record, (centers, labels, np.asarray(details["objectives"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spherical clustering, representatives, outliers, and MPI scaling")
    add_common_arguments(parser, "local")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--outlier-fraction", type=float, default=0.03)
    parser.add_argument("--input-file", help="Float32-compatible .npy embedding matrix")
    parser.add_argument("--initial-file", help="Optional .npy initial centers; same normalized embedding space")
    parser.add_argument("--mpi-ranks", type=int, nargs="*", default=[])
    parser.add_argument("--mpi-backend", choices=["cpu", "openmp", "cuda-naive", "cuda-optimized", "cublas"], default="openmp")
    parser.add_argument("--mpi-hosts", help="Optional Open MPI host list; defaults to local rank testing")
    parser.add_argument("--mpi-timeout", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true", help="Write input and initial centers without running benchmarks")
    args = parser.parse_args(argv)
    validate_common(args)
    if args.iterations < 1 or args.mpi_timeout < 1 or any(r < 1 for r in args.mpi_ranks):
        parser.error("iterations, MPI timeout and rank counts must be positive")
    if args.input_file:
        x, truth, anomalous = load_matrix(args.input_file), None, None
    else:
        x, truth, anomalous = cluster_demo(args.rows, args.dim, args.clusters, args.outlier_fraction, args.seed)
    initial = load_matrix(args.initial_file) if args.initial_file else initial_centers(x, args.clusters, args.seed)
    if initial.shape != (args.clusters, x.shape[1]):
        parser.error("Initial center shape must match clusters and embedding dimension")
    out = output_directory(args.output).resolve()
    x.tofile(out / "points.f32")
    initial.tofile(out / "initial.f32")
    np.save(out / "points.npy", x)
    np.save(out / "initial.npy", initial)
    write_json(out / "input.json", {"rows": len(x), "dim": x.shape[1], "clusters": args.clusters,
                                   "iterations": args.iterations, "seed": args.seed,
                                   "source": args.input_file or "synthetic spherical clusters with uniform outliers"})
    if truth is not None:
        np.savez(out / "ground_truth.npz", labels=truth, anomalous=anomalous)
    if args.prepare_only:
        print("Prepared input:", out)
        return
    reference, _, ref_objectives = kmeans(x, initial, args.iterations, "cpu", args.threads)
    records, histories = [], {}

    def save_result(backend, record, result):
        centers, labels, objectives = result
        metrics, distances, ranking = quality(x, centers, labels, objectives, truth, anomalous)
        record.update(metrics)
        record["objective_relative_difference_vs_cpu"] = float(abs(objectives[-1]-ref_objectives[-1])/max(1, abs(ref_objectives[-1])))
        np.savez(out / f"{backend}.npz", centers=centers, labels=labels, objectives=objectives)
        reports(out, backend, centers, labels, x, distances, ranking)
        histories[backend] = objectives
        return record

    for backend in dict.fromkeys(args.backends):
        ok, reason = availability(backend)
        if not ok:
            records.append({"backend": backend, "status": "skipped", "reason": reason})
            print(f"{backend}: skipped ({reason})", flush=True)
            continue
        try:
            result, elapsed = benchmark(lambda: kmeans(x, initial, args.iterations, backend, args.threads), backend, args.repeats)
            record = save_result(backend, {"backend": backend, "status": "ok", **elapsed}, result)
            records.append(record)
            print(f"{backend}: {elapsed['median_ms']:.3f} ms; objective={record['objective']:.4f}", flush=True)
        except Exception as exc:
            records.append({"backend": backend, "status": "error", "reason": str(exc)})
            print(f"{backend}: ERROR {exc}", flush=True)
    for ranks in dict.fromkeys(args.mpi_ranks):
        record, result = run_mpi(out, args, x, args.mpi_backend, ranks)
        if result is not None:
            try:
                record = save_result(record["backend"], record, result)
            except Exception as exc:
                record.update(status="error", reason=str(exc))
        records.append(record)
        print(f"{record['backend']}: {record['status']}", flush=True)
    write_json(out / "results.json", {"project": "distributed-clustering", "environment": metadata(),
               "rows": len(x), "dim": x.shape[1], "clusters": args.clusters, "iterations": args.iterations,
               "seed": args.seed, "threads_per_worker": args.threads,
               "timing_scope": "Single-process warm wall time includes allocation, transfers and objective evaluation. "
                               "MPI median includes process launch, input distribution and output I/O; compute-loop "
                               "and collective fields are separate timings from the final repeat. Rank timing maxima "
                               "can occur on different ranks and are not additive. Local ranks do not establish multi-node scaling.",
               "algorithm": "Fixed-iteration spherical k-means; empty or zero-sum clusters retain previous centers. "
                            "Reported final labels are reassigned to final centers. All backends share the same initialization.",
               "records": records})
    write_csv(out / "timings.csv", records)
    timing_plot(out / "clustering_latency.png", records, "Clustering: warm wall-clock latency")
    plot_objectives(out / "convergence.png", histories)
    print("Results:", out)
    if not any(r["status"] == "ok" for r in records) or any(r["status"] == "error" for r in records):
        raise SystemExit(1)
