"""Run independent frozen replay cases with explicit logs and bounded workers."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    if args.cases:
        cases = {name: cases[name] for name in args.cases}
    if not 1 <= args.jobs <= 4:
        raise ValueError("jobs must be between one and four")
    args.output_root.mkdir(parents=True, exist_ok=True)

    def execute(name, case):
        output = args.output_root / name
        if output.exists():
            raise ValueError(f"choose a fresh case output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("slam_loop_graph.py")),
            "--local-run",
            case["local_run"],
            "--output-dir",
            str(output),
        ]
        if case.get("features"):
            command.extend(["--features", case["features"]])
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            )
        return {"returncode": process.returncode, "output": str(output)}

    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {
            pool.submit(execute, name, case): name for name, case in cases.items()
        }
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = future.result()
            except (OSError, ValueError, KeyError) as error:
                results[name] = {"error": str(error), "returncode": 1}
            print(name, results[name], flush=True)
            (args.output_root / "batch_status.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    if any(r["returncode"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
