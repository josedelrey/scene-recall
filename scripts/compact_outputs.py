"""Archive compact research evidence, verify it, then discard bulky outputs.

Preview by default. --apply removes original output files only after every
retained byte is verified in an installed archive. Source datasets outside
outputs and versioned project files are never candidates.
"""

import argparse
import gzip
import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

TEXT = {".json", ".txt", ".csv", ".py", ".log", ".svg"}
CORE = ("trajectory", "trajectories", "measurement", "factors", "tracks", "poses")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def snapshot(root, *, hashes=False):
    if root.name != "outputs" or root.is_symlink() or not root.is_dir():
        raise ValueError("expected a regular outputs directory")
    if (root / "evidence").exists():
        raise ValueError("evidence capsule already exists, refusing to overwrite it")
    records = []
    for path in sorted(root.rglob("*")):
        s = path.lstat()
        if stat.S_ISDIR(s.st_mode):
            continue
        if not stat.S_ISREG(s.st_mode):
            raise ValueError(f"refusing non-regular output: {path}")
        relative = path.relative_to(root.parent)
        keep = path.suffix in TEXT or (
            path.suffix == ".npz" and any(word in path.stem for word in CORE)
        )
        record = {
            "path": relative.as_posix(),
            "bytes": s.st_size,
            "allocated_bytes": getattr(s, "st_blocks", (s.st_size + 511) // 512) * 512,
            "signature": signature(path),
            "retained": keep,
            "reason": "report, provenance, source snapshot or scientific trace"
            if keep
            else "rebuildable dataset copy, cache, geometry, prediction or visualization",
        }
        if hashes:
            record["sha256"] = digest(path)
            if signature(path) != record["signature"]:
                raise ValueError(f"output changed while hashing: {path}")
        records.append(record)
    return records


def verify_archive(path, expected):
    found = set()
    with tarfile.open(path, "r|xz") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in found
            ):
                raise ValueError("unexpected archive member")
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected[member.name]:
                raise ValueError(f"archive checksum differs: {member.name}")
            found.add(member.name)
    if found != set(expected):
        raise ValueError("archive omitted retained evidence")


def compact(root):
    if root.is_symlink():
        raise ValueError("refusing symlinked outputs directory")
    root = root.resolve()
    records = snapshot(root, hashes=True)
    groups = defaultdict(list)
    for record in records:
        if record["retained"]:
            parts = Path(record["path"]).parts
            group = parts[1] if len(parts) > 2 else "_root"
            groups[group].append(record)
    archives = {}
    print(
        f"Checked {len(records):,} original files, archiving {sum(map(len, groups.values())):,}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        prefix=".outputs-compaction-", dir=root.parent
    ) as temporary:
        capsule = Path(temporary) / "evidence"
        capsule.mkdir()
        for group, members in sorted(groups.items()):
            name = group + ".tar.xz"
            target = capsule / name
            with tarfile.open(target, "w:xz", preset=6) as archive:
                for record in members:
                    path = root.parent / record["path"]
                    info = archive.gettarinfo(str(path), arcname=record["path"])
                    # Regular members permit selective restoration even when
                    # original datasets contained hard-linked metadata tables.
                    info.type, info.linkname = tarfile.REGTYPE, ""
                    info.size = record["bytes"]
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
            verify_archive(target, {r["path"]: r["sha256"] for r in members})
            # Windows requires write access to flush file buffers.
            with target.open("r+b") as stream:
                os.fsync(stream.fileno())
            archives[name] = {
                "sha256": digest(target),
                "bytes": target.stat().st_size,
                "members": len(members),
            }
            print(
                f"Verified {name}: {target.stat().st_size / 2**20:.1f} MiB", flush=True
            )
        inventory = capsule / "inventory.jsonl.gz"
        with gzip.open(inventory, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        manifest = {
            "format_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "archives verified before original removal",
            "original_files": len(records),
            "retained_files": sum(r["retained"] for r in records),
            "original_logical_bytes": sum(r["bytes"] for r in records),
            "retained_uncompressed_bytes": sum(
                r["bytes"] for r in records if r["retained"]
            ),
            "inventory_sha256": digest(inventory),
            "archives": archives,
            "scope": "Original outputs only. External datasets, environment and versioned source untouched.",
            "historical_manifests": "Preserved byte-for-byte. They describe pre-cleanup files, including intentionally discarded artifacts.",
            "restore": "Run tar -xJf outputs/evidence/STUDY.tar.xz -C . from the repository. Select individual member paths to restore only needed evidence.",
            "rerun": "Restoration supplies reports and traces, not deleted images, feature caches or map predictions. Reacquire inputs and regenerate those products before old verification or mapping commands.",
        }
        (capsule / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in (inventory, capsule / "manifest.json"):
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        sync_directory(capsule)
        # Check the complete snapshot before the first unlink. Hard-link ctimes
        # can change during removal, so this check deliberately precedes it.
        current = {r["path"]: r["signature"] for r in snapshot(root)}
        if current != {r["path"]: r["signature"] for r in records}:
            raise ValueError("outputs changed, originals preserved")
        os.replace(capsule, root / "evidence")
        sync_directory(root)
        for record in records:
            (root.parent / record["path"]).unlink()
        for directory in sorted(
            root.rglob("*"), key=lambda p: len(p.parts), reverse=True
        ):
            if directory.is_dir() and not directory.is_relative_to(root / "evidence"):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        manifest["status"] = "originals removed after archive verification"
        receipt = root / "evidence/manifest.json"
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")
        with receipt.open("r+b") as stream:
            os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="archive, verify and remove originals"
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "outputs"], cwd=repository, text=True
    )
    if tracked.strip():
        parser.error("outputs contains versioned files")
    root = repository / "outputs"
    if args.apply:
        compact(root)
    else:
        records = snapshot(root)
        kept = [r for r in records if r["retained"]]
        print(
            json.dumps(
                {
                    "files": len(records),
                    "logical_bytes": sum(r["bytes"] for r in records),
                    "retain_files": len(kept),
                    "retain_uncompressed_bytes": sum(r["bytes"] for r in kept),
                    "apply": "pass --apply to archive, verify and delete originals",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
