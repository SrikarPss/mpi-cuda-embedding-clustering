# Standalone verification

This repository was copied by itself into a fresh temporary directory, excluding build outputs, results, caches, virtual environments, and sibling sources. Source/build override environment variables were cleared. The path included spaces to exercise command/path handling.

| Check | Result |
|---|---|
| CMake configure and native CPU/OpenMP compilation | Passed |
| Numerical and real-input CLI tests | 5 passed; 1 skipped |
| CPU/OpenMP/PyTorch CPU synthetic demo | Passed |
| Size/thread scaling script | Passed |
| Markdown report generation | Passed |
| Git excludes build, results, caches, and environments | Passed |

The verification used an existing Python 3.13 interpreter with this repository's third-party dependencies installed. No parent or sibling project code was present in the source copy. A fresh checkout can install the dependencies into its own virtual environment using the README instructions.

CUDA was disabled for this macOS build; CUDA source still requires NVIDIA compilation and execution. MPS execution needs a regular terminal with Metal access.

The single-process MPI test passed. The multi-rank MPI test is opt-in because this execution session denies launcher socket access. Enable it with `HPC_TEST_MPI=1 python -m unittest discover -s tests -v` in a regular terminal.
