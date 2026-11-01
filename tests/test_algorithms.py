"""Independent numerical references and CLI checks for distributed-clustering."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from distributed_clustering.algorithms import initial_centers, kmeans
from distributed_clustering.data import cluster_demo
from distributed_clustering.runtime import BACKENDS, availability, normalize


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)
        self.backends = [b for b in BACKENDS if availability(b)[0]]
        self.assertIn("cpu", self.backends, "Build the native library first")

    def test_spherical_kmeans_objective_and_final_assignment(self):
        x, truth, _ = cluster_demo(301, 17, 4, outlier_fraction=0, seed=12)
        initial = initial_centers(x, 4, seed=42)
        for backend in self.backends:
            with self.subTest(backend=backend):
                centers, labels, history = kmeans(x, initial, 8, backend)
                self.assertTrue(np.all(np.diff(history) < 1e-3))
                assigned = np.argmax(x.astype(np.float64) @ centers.astype(np.float64).T, axis=1)
                np.testing.assert_array_equal(labels, assigned)
                expected = np.sum(1-np.einsum("ij,ij->i", x, centers[labels]), dtype=np.float64)
                self.assertAlmostEqual(history[-1], expected, delta=1e-3)
                np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1, atol=2e-5)

    def test_duplicate_centers_and_zero_points(self):
        x = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], np.float32)
        initial = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
        for backend in self.backends:
            with self.subTest(backend=backend):
                centers, labels, _ = kmeans(x, initial, 3, backend)
                np.testing.assert_array_equal(labels, [0, 0, 2, 2])
                np.testing.assert_array_equal(centers, initial)
                centers, labels, history = kmeans(np.zeros((4, 2)), np.zeros((2, 2)), 2, backend)
                np.testing.assert_array_equal(labels, [0, 0, 0, 0])
                np.testing.assert_array_equal(history, [4, 4, 4])

    def test_mpi_single_process_matches_native_reference(self):
        executable = ROOT / "build/hpc_cluster_mpi"
        if not executable.exists():
            self.skipTest("MPI target was not built")
        with tempfile.TemporaryDirectory(prefix="hpc-mpi-single-") as directory:
            out = Path(directory)
            x, _, _ = cluster_demo(31, 7, 3, 0, 8)
            initial = initial_centers(x, 3)
            expected_c, expected_l, expected_h = kmeans(x, initial, 4)
            x.tofile(out / "points.f32")
            initial.tofile(out / "initial.f32")
            backend = "openmp" if availability("openmp")[0] else "cpu"
            command = [str(executable), "--input", str(out / "points.f32"), "--initial", str(out / "initial.f32"),
                       "--rows", "31", "--dim", "7", "--clusters", "3", "--iterations", "4", "--threads", "2",
                       "--backend", backend, "--output", str(out / "result")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            np.testing.assert_allclose(np.fromfile(out / "result.centers.f32", np.float32).reshape(3, 7), expected_c, atol=2e-5)
            np.testing.assert_array_equal(np.fromfile(out / "result.labels.i32", np.int32), expected_l)
            import json
            history = json.loads((out / "result.json").read_text())["objectives"]
            np.testing.assert_allclose(history, expected_h, rtol=2e-5, atol=2e-5)

    @unittest.skipUnless(os.environ.get("HPC_TEST_MPI") == "1", "Set HPC_TEST_MPI=1 in a terminal to test MPI launches")
    def test_mpi_uneven_and_empty_partitions(self):
        for n, ranks in [(17, 3), (2, 3)]:
            with self.subTest(rows=n, ranks=ranks), tempfile.TemporaryDirectory(prefix="hpc-mpi-test-") as directory:
                out = Path(directory)
                x, _, _ = cluster_demo(n, 7, 2, 0, 8)
                initial = initial_centers(x, 2)
                expected_c, expected_l, _ = kmeans(x, initial, 3)
                x.tofile(out / "points.f32")
                initial.tofile(out / "initial.f32")
                env = dict(os.environ, PATH="/opt/homebrew/bin:"+os.environ.get("PATH", ""))
                command = ["mpirun", "--oversubscribe", "-n", str(ranks), str(ROOT / "build/hpc_cluster_mpi"),
                           "--input", str(out / "points.f32"), "--initial", str(out / "initial.f32"),
                           "--rows", str(n), "--dim", "7", "--clusters", "2", "--iterations", "3", "--threads", "2",
                           "--backend", "openmp", "--output", str(out / "result")]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                np.testing.assert_allclose(np.fromfile(out / "result.centers.f32", np.float32).reshape(2, 7), expected_c, atol=2e-5)
                np.testing.assert_array_equal(np.fromfile(out / "result.labels.i32", np.int32), expected_l)

    def test_normalization_handles_zero_large_and_small_rows(self):
        np.testing.assert_allclose(normalize([[0, 0], [3, 4]]), [[0, 0], [0.6, 0.8]], atol=2e-6)
        np.testing.assert_allclose(normalize([[1e30, 1e30], [1e-40, 0]]),
                                   [[2**-0.5, 2**-0.5], [1, 0]], atol=2e-6)

    def test_cli_real_data_input(self):
        import json
        with tempfile.TemporaryDirectory(prefix="standalone-cli-") as directory:
            root = Path(directory)
            x, _, _ = cluster_demo(97, 13, 4, 0, 7)
            np.save(root / "points.npy", x)
            np.save(root / "initial.npy", initial_centers(x, 4))
            arguments = ["--input-file", str(root / "points.npy"), "--initial-file", str(root / "initial.npy"),
                         "--clusters", "4", "--iterations", "3"]
            out = root / "output"
            command = [sys.executable, str(ROOT / "run.py"), "--backends", "cpu", "--repeats", "1",
                       "--output", str(out), *arguments]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads((out / "results.json").read_text())["records"][0]
            self.assertEqual(record["status"], "ok")
            self.assertNotIn("pair_precision", record)
            self.assertNotIn("inlier_label_accuracy_after_matching", record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
