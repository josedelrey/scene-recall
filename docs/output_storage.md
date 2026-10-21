# Research output storage and recovery

Generated research outputs were compacted on 2026-10-21 to reclaim disk space.
Versioned source, tests, documentation, dependencies and external datasets were
preserved. Historical experiment results remain documented in their study guides.

The allocated output footprint fell from **21.0 GiB to 395 MiB**, reclaiming
**20.6 GiB (98.2%)**. Twelve study archives retain 3,570 evidence files from
29,979 original files. The inventory fingerprints all originals. A post-cleanup
check verified every installed archive checksum and restored full native Desk,
RPY and XYZ trajectories with 573, 694 and 792 frames through the normal reader.
`outputs/evidence/cleanup_result.json` records the measured sizes and checks.

## Retained evidence

The local `outputs/evidence/` capsule contains one `.tar.xz` archive per study,
plus `manifest.json` and `inventory.jsonl.gz`. It is ignored by Git and does not
travel with the repository or private project context.

The archives preserve these original files byte-for-byte:

- JSON reports, results, frozen protocols, configurations and historical manifests
- Final and emitted trajectories, multiview tracks and measurement/factor packets
- Dataset timestamp/reference tables, preparation records and download hashes
- CSV tables, vector figures, diagnostic logs and historical Python source snapshots

The inventory records every original file's path, SHA256, size and retention
decision. The capsule manifest records archive checksums. Every retained file
was read back from its archive and verified before original files were removed.
Archive members are regular files with their original repository-relative paths,
including metadata that previously used hard links.

Large regenerable products were deleted: dataset archives and images inside
`outputs/`, feature/descriptor caches, voxel and TSDF volumes, point clouds,
meshes, rendered predictions and target arrays, Rerun recordings, raster/HTML
previews and build artifacts. There is no local backup of these deleted products.

Historical manifests describe the complete **pre-cleanup** experiments. Their
checksums were preserved without rewriting them to pretend deleted files remain
available. Old full verification commands require regenerating the missing
products. The capsule inventory is the record of what remains recoverable locally.

## Read or restore evidence

Run these examples from the repository root:

```bash
# Inspect the current capsule and archive contents.
cat outputs/evidence/manifest.json
tar -tJf outputs/evidence/slam_lifecycle.tar.xz

# Read a summary directly, without expanding a study.
tar -xJOf outputs/evidence/slam_lifecycle.tar.xz \
  outputs/slam_lifecycle/summary.json

# Restore just a saved trajectory and its report into their original paths.
tar -xJf outputs/evidence/slam_lifecycle.tar.xz -C . \
  outputs/slam_lifecycle/full/desk/native/report.json \
  outputs/slam_lifecycle/full/desk/native/trajectory.npz
```

Extract into paths that do not contain newer runs. To restore a complete study's
compact evidence, omit the member paths. This restores its reports and traces,
without restoring the deleted images, caches or geometry.

For ongoing SLAM work, the most useful archives are `slam_lifecycle.tar.xz`,
`incremental_slam.tar.xz`, `local_map_reliability.tar.xz`,
`local_map_refinement.tar.xz` and `local_map.tar.xz`. They contain the latest
experiments and their baseline lineage. Other study archives retain earlier
odometry, multiview, mapping, reconstruction and visualization evidence.

## Inputs needed for new experiments

External TUM Freiburg 1 XYZ, Desk and RPY datasets and the existing ScanNet
capture were outside `outputs/` and remain untouched. Their availability is
machine-specific. Supply explicit dataset paths as described in the dataset guides.

These downloaded datasets were removed from inside `outputs/`. Reacquire them
from the [official TUM dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
when needed, using the original archive names and checksums:

| Archive | SHA256 |
| --- | --- |
| `rgbd_dataset_freiburg2_desk.tgz` | `1a0756d72510a26e26e2a02bf0b6c69c2796619fca531e176fc2e5110173807c` |
| `rgbd_dataset_freiburg3_long_office_household.tgz` | `c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c` |
| `rgbd_dataset_freiburg3_structure_texture_far.tgz` | `3f58c707f54c93b68fecd77293630e96a76d7b0f2703eb8ad9cff45ba4bbb81a` |
| `rgbd_dataset_freiburg1_room.tgz` | `5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da` |

Original download records are archived at
`outputs/local_map_refinement/download_manifest.json`,
`outputs/incremental_slam/downloads/manifest.json` and
`outputs/slam_lifecycle/download.json`. The latter two include recorded direct URLs.
Inspect restored reports for source locations, calibration, canonical fingerprints
and frame ranges before rerunning a comparison.

Freiburg 2 Desk has duplicate reference timestamps. Recreate its separate prepared
copy with `experiments/prepare_tum_references.py`. This omits every ambiguous row,
hard-links images and leaves the original source unchanged. Prepare into a fresh
directory before restoring archived metadata there. Preserve the recorded table
hashes and do not silently relax the canonical reader.

Regenerate feature caches with `experiments/cache_slam_baseline.py`, or use the
research drivers' raw-extraction mode. Rebuild maps and predictions from restored
trajectories and original RGB-D, following each study's frozen configuration.
Recordings can then be exported again. Restore factor packets and protocols
alongside reports when doing source-free numerical replay. A rendered-map or
image-based experiment still needs its original raw inputs.

## Future cleanup

```bash
uv run python scripts/compact_outputs.py          # Preview only.
uv run python scripts/compact_outputs.py --apply  # Archive, verify, then remove originals.
```

Stop output-producing jobs first. The tool refuses tracked output files,
symlinks, an existing evidence capsule and changes detected during archiving.
It preserves the capsule before deleting originals. An existing capsule must be
handled explicitly before another compaction, so it cannot be silently replaced.
For routine work, generate only required exports and discard rebuildable products
after retaining the final reports, trajectories and measurement provenance.
