# Initial TUM and cross-dataset odometry evaluation

October 10, 2026. TUM `freiburg1_xyz` now runs through SceneRecall's canonical
RGB-D data, all three existing odometry backends, and the shared evaluation and
export pipeline. The adapter supplies independent motion-capture references.
Backend settings and acceptance policies are unchanged.

The subsequent [cross-sequence evaluation](tum_cross_sequence_evaluation.md)
covers the two planned Freiburg 1 motion and office-sweep sequences and
reassesses the development path toward multiframe mapping.

## Method and source validation

The first experiment evaluates the complete associated sequence. The source
contains 798 RGB images, 798 depth images, and 3000 ground-truth samples.
One-to-one association within 20 ms produces 792 observations over 26.594239
seconds, leaving six unmatched images in each stream. Median absolute RGB/depth
skew is 7.320 ms and maximum skew is 17.230 ms. Median consecutive depth interval
is 32.548 ms and the maximum is 66.331 ms. These gaps are preserved.

789 observations have usable interpolated ground truth. Associated frames
195–197 cross a 110.100 ms ground-truth gap whose surrounding samples exceed
the 50 ms interpolation tolerance. These observations remain available to odometry.
Accuracy metrics exclude their references. The final associated frame, index
791, has a valid reference and supports endpoint evaluation.

The registered-image calibration uses TUM's recommended ROS-default pinhole
policy. PNG depth is divided by 5000 with no further correction. Camera-to-world
poses are evaluated at depth timestamps with bounded linear translation and
shortest-arc quaternion SLERP. The complete parsing, synchronization, calibration
limitations, and coordinate conventions are described in [TUM support](tum.md),
with links to the official dataset documentation.

Runs execute serially using one OpenCV, OpenBLAS, and OMP thread. All three
backends see identical canonical inputs, references, timestamps, and calibration.
The comparison verifies exact component settings and package fingerprints.
A single run provides descriptive throughput. Cache and system load still affect
these measurements, and throughput does not establish real-time operation.

The default frame RPE intervals remain 1, 5, and 10. Time RPE uses 0.1, 0.5, and
1.0 seconds, with the closest later endpoint within 20 ms and actual elapsed-time
distributions recorded. Comparison errors use identical reference-valid pairs
that are continuous in every backend. Each run additionally preserves its own
accuracy population and per-segment metrics. A tracking loss starts a new local
segment, without reference-based correction or stitching.

## Results

All three backends were evaluated on the same 792 observations and 791
attempted transitions. Sparse and hybrid retained one continuous trajectory.
ICP tracked 761 transitions and lost 30, all at the finest-scale iteration
limit. Its largest segment contains 94 frames. The default hybrid policy
retained sparse motion on 29 ICP iteration-limit failures and refined 762 pairs.
Every hybrid sparse initialization equals the standalone sparse pose exactly.

| Backend | Tracking coverage | Segments | Common RPE 1 mm / deg | Common RPE 10 mm / deg | Aligned ATE mm | Anchored RMSE mm | Endpoint mm / deg | Frames/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse | 100.00% | 1 | 5.299 / 0.301 | 17.168 / 0.864 | 55.994 | 101.797 | 140.110 / 2.499 | 4.24 |
| icp | 96.21% | 31 | 2.778 / 0.447 | 13.069 / 1.283 | — | — | — | 1.76 |
| hybrid | 100.00% | 1 | 2.785 / 0.445 | 12.977 / 1.288 | 25.514 | 43.161 | 64.746 / 3.403 | 1.31 |

The common frame RPE populations contain 757, 644, and 527 pairs at intervals
1, 5, and 10. A dash denotes unavailable full-sequence metrics after tracking
losses. ICP retains its segment-level errors in the report. These short segments
must not be treated as a continuous full-sequence trajectory.

| Time interval | Common pairs | Sparse mm / deg | ICP mm / deg | Hybrid mm / deg | Actual interval mean / max seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 s | 693 | 8.114 / 0.428 | 4.934 / 0.657 | 4.961 / 0.658 | 0.100086 / 0.106268 |
| 0.5 s | 422 | 23.039 / 1.165 | 18.409 / 1.659 | 18.186 / 1.666 | 0.500517 / 0.507226 |
| 1.0 s | 232 | 40.497 / 2.049 | 31.287 / 2.516 | 30.935 / 2.531 | 1.001060 / 1.008338 |

Hybrid improves translation at every matched frame and time horizon, and
reduces aligned position ATE from 55.994 to 25.514 mm. Sparse has lower rotation
RPE at every matched horizon. Hybrid endpoint rotation rises from 2.499 to
3.403 degrees despite its lower translation drift. These outcomes support
evaluating translation, rotation, continuity, and runtime separately.

Of the 762 successful hybrid refinements, 758 have valid reference pairs.
Translation error improves on 578/758 (76.25%) and rotation improves on 204/758
(26.91%). On that same population, translation RMSE falls from 5.298 to 2.779 mm,
while rotation RMSE rises from 0.301 to 0.447 degrees. Common held-out depth
residual decreases on 756/758 pairs, including 180 with worse translation and
553 with worse rotation. Lower depth residual therefore remains insufficient
to establish improved pose accuracy.

![TUM matched backend errors, endpoint drift, and throughput](/home/rlyeh/repos/scene-recall/outputs/odometry/tum_freiburg1_xyz/backend_comparison.png)

![Hybrid depth residual and independent reference error changes](/home/rlyeh/repos/scene-recall/outputs/odometry/tum_freiburg1_xyz/refinement_tradeoff.png)

## Cross-dataset context

The previous [ScanNet matrix](odometry_backend_comparison.md) covers five
200-frame clips from `scene0000_00`. The following table uses the same error
definitions and fixed default backend settings, with each dataset retaining its
own common population. It compares observed behavior across these captures,
rather than claiming equal motion difficulty or equal reference quality.

| Dataset and sample | Backend | Tracking coverage | Common RPE 1 mm / deg | Common RPE 10 mm / deg |
| --- | --- | ---: | ---: | ---: |
| ScanNet, 5 × 200 frames | sparse | 100.00% | 8.621 / 0.222 | 23.721 / 0.580 |
| ScanNet, 5 × 200 frames | icp | 99.20% | 9.841 / 0.253 | 52.799 / 1.574 |
| ScanNet, 5 × 200 frames | hybrid | 100.00% | 9.298 / 0.253 | 51.163 / 1.568 |
| TUM xyz, 792 frames | sparse | 100.00% | 5.299 / 0.301 | 17.168 / 0.864 |
| TUM xyz, 792 frames | icp | 96.21% | 2.778 / 0.447 | 13.069 / 1.283 |
| TUM xyz, 792 frames | hybrid | 100.00% | 2.785 / 0.445 | 12.977 / 1.288 |

Hybrid translation gains on TUM contrast with the weaker aggregate translation
results in the prior ScanNet clips. The rotation trade-off persists here against
independent ground truth. This extends the earlier observation that a smaller
depth residual does not guarantee a better pose in every dimension. It does
not isolate calibration, motion, scene geometry, or reconstruction-reference
quality as the cause of the dataset difference.

The ScanNet sample has no usable timestamps, so elapsed-time RPE remains
unavailable there. Its reconstruction references are coupled to scene geometry,
while TUM uses independent motion capture. Avoid pooling both sources into
one accuracy number or treating one xyz sequence as representative.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python experiments/compare_odometry_backends.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_xyz \
  --ranges all --output-dir outputs/odometry/tum_xyz_repeat
```

Measured runs are in `outputs/odometry/tum_freiburg1_xyz/`. Each backend produces
`report.json`, `trajectory.npz`, and `frames.csv`. The root `comparison.json`
records common pair populations, failure reasons, refinement outcomes,
configuration, timing, and report hashes. All generated data and outputs remain
ignored by Git. The original ScanNet outputs are preserved.

Optional figures use the existing dataset-aware plotting script without adding
a project dependency:

```bash
MPLCONFIGDIR=/tmp/scene-recall-mpl uv run --with matplotlib==3.11.2 python \
  experiments/plot_odometry_backends.py outputs/odometry/tum_freiburg1_xyz
```

## Verification

All 475 tests pass. The added tests cover registered calibration, metric z-depth,
RGB channel order, unique timestamp association, source range semantics,
quaternion order and normalization, SLERP, transform direction, missing ground
truth, unsupported ground-truth gaps, malformed input, reader closure, all three
real backends through TUM, reference independence, timestamp RPE, and mixed
comparison dispatch. Lint, formatting, and whitespace checks pass.

All three backends were repeated on associated TUM frames `[0, 20)`. All 12
archive arrays match their full-sequence prefixes exactly, and numerical frame
diagnostics match after excluding stage runtime. TUM package fingerprints match
between full runs and repeats. Settings and calibration are identical.

Fresh ScanNet `[0, 20)` runs through the shared runner also match all 12 archive
arrays and numerical diagnostics in the historical `[0, 200)` matrix prefixes.
This verifies that the runner refactor preserves the existing estimates and
failure semantics on that sample. It does not substitute for evaluating new
ScanNet scenes. These repeats are stored in `outputs/odometry/tum_xyz_repeat20/`
and `outputs/odometry/scannet_shared_repeat20/`. The checks are recorded in
`outputs/odometry/tum_freiburg1_xyz/reproducibility.json`. Source table hashes are
in `source_manifest.json` beside the full comparison.

PNG and SVG figures were rendered with Matplotlib 3.11.2 and inspected visually.
No dependencies were added and no commit was created.

## Interpretation limits and next experiments

This validation extends evaluation to a second dataset and independently
measured poses. One TUM sequence and one ScanNet capture do not establish broad
generalization. `freiburg1_xyz` is an initial translation-oriented validation
case, rather than a representative average over motion and scene conditions.
The official [sequence descriptions](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
provide the basis for the next selection:

1. Run `freiburg1_rpy` to isolate rotation-dominated motion and
   `freiburg1_desk` / `desk2` for office sweeps with combined motion. Freeze
   backend settings and association policy before evaluating them.
2. Run `freiburg2_xyz` and `freiburg2_rpy` for a different calibrated Kinect and
   slower, longer trajectories. Validate source conventions before extending to
   Freiburg 3's Asus Xtion sequences, then compare a static scene with
   `freiburg3_walking_xyz` to expose moving-object sensitivity.
3. Diagnose the hybrid objective using successful refinements on identical pairs:
   reference error, held-out depth residual, correction magnitude, support, and
   convergence. Keep settings fixed on evaluation sequences. If tuning is
   justified, use separate development sequences.
4. Measure sensitivity to RGB/depth skew and ground-truth association limits.
   A separately validated sensor-specific rectification policy would help bound
   the remaining calibration approximation. Avoid silently substituting measured
   distorted RGB intrinsics into the current pinhole contract.
5. Expand ScanNet beyond `scene0000_00`. Report sequence-level distributions of
   tracking coverage, uninterrupted duration, ATE, and matched time RPE where
   clocks are available. Keep reconstruction references and motion-capture
   references identified separately.
