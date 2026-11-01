#!/usr/bin/env python3
"""Configure and compile native CPU/OpenMP, MPI, and optional CUDA targets."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--no-mpi", action="store_true")
parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
parser.add_argument("--cuda-arch", help="CMake CUDA architecture, e.g. 89; default: compiler default")
args = parser.parse_args()
env = os.environ.copy()
env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
cmake = shutil.which("cmake", path=env["PATH"])
if not cmake:
    parser.error("CMake missing. macOS: brew install cmake ninja libomp open-mpi")
command = [cmake, "-S", str(ROOT), "-B", str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
           "-DHPC_PYTHON_EXECUTABLE=" + sys.executable,
           "-DHPC_ENABLE_CUDA=" + ("OFF" if args.no_cuda else "ON"),
           "-DHPC_ENABLE_MPI=" + ("OFF" if args.no_mpi else "ON")]
sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
if sys.platform == "darwin" and sdk.exists():
    command.append("-DCMAKE_OSX_SYSROOT=" + str(sdk))
    env["SDKROOT"] = str(sdk)
if args.cuda_arch:
    command.append("-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch)
subprocess.run(command, env=env, check=True)
subprocess.run([cmake, "--build", str(args.build_dir), "--parallel", str(min(4, os.cpu_count() or 1))], env=env, check=True)
print("Build complete:", args.build_dir)
