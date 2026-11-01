#!/usr/bin/env python3
"""Generate a Markdown report from this repository's saved measurements."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/local")
args = parser.parse_args()
data = json.loads((args.directory / "results.json").read_text())
lines = ["# Benchmark report", "", data["timing_scope"], "", "| Backend | Status | Median ms |", "|---|---|---:|"]
for row in data["records"]:
    elapsed = f"{row['median_ms']:.3f}" if row["status"] == "ok" else "—"
    lines.append(f"| {row['backend']} | {row['status']} | {elapsed} |")
lines += ["", "Full samples, quality metrics, configuration, and backend reasons are in results.json.", ""]
path = args.directory / "REPORT.md"
path.write_text("\n".join(lines))
print(path)
