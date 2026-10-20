"""Prepare a separate TUM copy omitting all ambiguous reference timestamps.

The strict canonical reader stays unchanged. Original archive contents remain
untouched. Image files are hard-linked, and the reference table alone is replaced.
No pose is selected from conflicting rows and no reference sample is invented.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from itertools import pairwise
from pathlib import Path

from scene_recall.run_metadata import sha256


def prepare(source, output):
    source = source.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("choose a fresh prepared source")
    table = source / "groundtruth.txt"
    lines = table.read_text(encoding="utf-8").splitlines(keepends=True)
    stamps = [
        float(line.split()[0])
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(b < a for a, b in pairwise(stamps)):
        raise ValueError("out-of-order reference samples require a separate audit")
    counts = Counter(stamps)
    ambiguous = {stamp for stamp, count in counts.items() if count > 1}
    if not ambiguous:
        raise ValueError("reference table has no duplicate timestamps")
    keep = [
        line
        for line in lines
        if not line.strip()
        or line.lstrip().startswith("#")
        or float(line.split()[0]) not in ambiguous
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output, copy_function=os.link)
    # Unlink first because the prepared table initially shares the source inode.
    (output / "groundtruth.txt").unlink()
    (output / "groundtruth.txt").write_text("".join(keep), encoding="utf-8")
    record = {
        "original_source": str(source),
        "original_table_sha256": sha256(table),
        "prepared_table_sha256": sha256(output / "groundtruth.txt"),
        "removed_timestamps_s": sorted(ambiguous),
        "removed_rows": sum(counts[t] for t in ambiguous),
        "policy": "omit every reference row at duplicated timestamps, preserve images and all remaining rows",
    }
    (output / "reference_preparation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
