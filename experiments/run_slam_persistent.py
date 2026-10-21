"""Run a frozen manifest of persistence-admission experiments in separate processes."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from scene_recall.run_metadata import code_hash, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON list of [name, baseline, features or raw]"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    if args.workers < 1 or args.output_dir.exists() or not cases:
        parser.error("provide positive workers, cases and a fresh output directory")
    if len({c[0] for c in cases}) != len(cases) or any(
        Path(c[0]).name != c[0] for c in cases
    ):
        parser.error("case names must be unique directory names")
    args.output_dir.mkdir(parents=True)
    protocol = {
        "phase": "frozen before evaluation",
        "cases": cases,
        "variants": ["persistent", "persistent_turnover"],
        "implementation": code_hash(),
        "driver_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.manifest),
        "workers": args.workers,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )

    def run(case):
        name, baseline, features = case
        command = [
            sys.executable,
            str(Path(__file__).with_name("slam_persistent.py")),
            "--baseline-run",
            baseline,
            "--output-dir",
            str(args.output_dir / name),
        ]
        if features != "raw":
            command += ["--features", features]
        with (args.output_dir / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return name, result.returncode

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, status in pool.map(run, cases):
            print(name, status, flush=True)
            if status:
                failed.append(name)
    if failed:
        raise RuntimeError("cases failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
