# RGB-D odometry comparison workflow

Compare `sparse`, `icp` and `hybrid` using the same canonical observations,
tracking, [evaluation conventions](odometry.md#evaluation) and version-2 exports.
The runner executes serially with one OpenCV thread and rotates backend order
across ranges. Keep BLAS and OMP at one thread for controlled comparisons.

## Run a matched comparison

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/compare_odometry_backends.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_desk \
  --ranges all --output-dir outputs/odometry/desk_comparison
```

`--sens-path` accepts one or more ScanNet captures. `--tum-path` accepts one or
more extracted TUM directories. Both can appear in one invocation. Use
`--ranges 0:200 1000:1200` for explicit half-open clips or `--ranges all` for
complete sequences. With no ranges, ScanNet-only runs use five 200-frame clips
starting at 0, 1000, 2000, 3500 and 4500. Any TUM input changes the default to
full sequences. Inputs must be long enough for each requested range.

Choose a fresh output directory. Individual subdirectories are named
`<sequence_id>_<start>_<stop-or-all>_<backend>` and contain `trajectory.npz`,
`frames.csv` and `report.json`. `comparison.json` records common pair counts,
errors, refinement outcomes, failures, timing, configuration and report/script
hashes. To recompute the summary from existing runs, use the same source/range
arguments and output directory with `--summarize-only`.

The comparison checks source selection, canonical input fingerprints,
calibration, timestamps, reference arrays, component settings, package identity
and numerical environment. Sparse and ICP component settings must match their
Hybrid counterparts. Keep settings fixed when evaluating held-out sequences.

## Interpret the comparison

Common RPE uses identical reference-valid pairs continuous in every backend.
Also report each backend's own population, tracking coverage, segment count and
largest segment. Fragmentation can make conditional accuracy appear favorable
by excluding difficult motion. Full-sequence ATE and endpoints require a
continuous trajectory. Never stitch independent origins with reference poses.

Hybrid comparisons assess accepted Sparse initialization and successful ICP
refinement on identical pairs. Fallbacks remain in tracking statistics and are
excluded from refinement improvement counts. Held-out depth scores use identical
source IDs with pose-specific associations. Target normal neighborhoods may
share fitting data, so these scores do not independently validate the sensors.

Translation, rotation, anchored drift, fixed-scale position ATE and runtime
answer different questions. A reduced depth residual, full local rank or large
support does not establish more accurate motion. Compare [map diagnostics](rgbd_mapping.md)
and [held-out fusion views](heldout_mapping_validation.md) when assessing the
consequences for reconstruction. ScanNet reconstruction references and TUM
motion capture have different error sources. Keep sequence results separate.
Throughput includes source opening, indexing, decoding and estimation, excluding
evaluation and export. Single-run timings depend on hardware, cache and load.

## Analyze and verify saved runs

```bash
uv run --locked python experiments/analyze_odometry_sequences.py \
  outputs/odometry/desk_comparison \
  --output-dir outputs/odometry/desk_analysis
```

Multiple comparison directories can be supplied. Original source datasets must
remain accessible for the TUM clock-sensitivity analysis. The analysis verifies report
hashes and backend settings and writes `analysis.json` and `summary.csv`. It
includes motion, tracking gaps, matched Sparse/Hybrid cohorts, longer horizons,
refinement outcomes, stage timing and RGB/depth-clock sensitivity. Changing the
evaluation clock does not correct the estimator's mixed-time measurements.

Repeat prefixes and later clips into a fresh directory, then verify against the
full comparison:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/compare_odometry_backends.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_desk \
  --ranges 0:20 100:125 --output-dir outputs/odometry/desk_repeats

uv run --locked python experiments/verify_odometry_repeats.py \
  --full-dir outputs/odometry/desk_comparison \
  --repeat-dir outputs/odometry/desk_repeats --hash-sources
```

The verifier checks configuration, calibration, software fingerprints, provenance,
segment boundaries and numerical diagnostics excluding runtime. Prefix archive
arrays must match exactly. Later clips compare pair estimates and estimated
trajectory coordinates after rebasing, without reference alignment. It writes
`reproducibility.json` and, with `--hash-sources`, `source_manifest.json`.

Optional plots require Matplotlib, which is outside the project dependencies:

```bash
uv run --with matplotlib==3.11.2 python experiments/plot_odometry_backends.py \
  outputs/odometry/desk_comparison
```

The sequence analysis command also accepts `--plot` when Matplotlib is available.

## Diagnostic utilities

Use `uv run --locked python <script> --help` for required inputs and options.
These utilities write generated data to ignored `outputs/`. Preserve original
runs and select fresh output paths.

| Script in `experiments/` | Purpose and prerequisites |
| --- | --- |
| `odometry_failure_probe.py` | Inject black RGB and missing depth into a 20-frame ScanNet clip, reporting losses and reset boundaries |
| `diagnose_odometry.py` | Inspect exact sparse 3D–2D inputs, shared residual populations and reference disagreement, requiring a saved default baseline through `--baseline-dir` |
| `depth_registration.py` | Compare independent bidirectional depth fits from sparse/reference initializations, requiring `.sens` input and saved `--runs` covering the target indices |
| `plot_depth_registration.py` | Plot a saved depth-registration report using optional Matplotlib |

The image diagnostic assumes the original ScanNet capture's RGB resolution.
Depth registration retains convergence traces and permissive candidates for
inspection. Production ICP applies stricter acceptance gates. The legacy
[two-frame SIFT utility](scannet.md#two-frame-sift-experiment) has its own saved
array schema and diagnostic-only LM option.
