#!/usr/bin/env python3
"""Distributed GPU Embedding Clustering and Outlier Discovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from distributed_clustering.app import main

if __name__ == "__main__":
    main()
