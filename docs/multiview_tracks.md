# Multiview RGB-D tracks

SceneRecall can extract persistent feature identities and shared metric landmarks
on an existing estimated trajectory. This is an offline measurement builder for
bounded trajectory ranges. It does not correct poses, perform Bundle Adjustment,
relocalize cameras, join lost segments or alter mapping.

## Run and reproduce

Use a saved format-2 odometry run and a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  scripts/rgbd_tracks.py \
  --odometry-run outputs/odometry/desk_sparse_200 \
  --output-dir outputs/multiview/desk --stride 2 --controls
```

The source is reopened through `datasets.replay`. Calibration, provenance,
timestamps and the full canonical input fingerprint are checked before publishing,
including views excluded by the feature stride. `--source` overrides the source
location subject to the existing replay checks. A segment root is always retained.
No reference pose enters descriptor matching, geometric gates or landmarks.
Reference arrays are compared only to verify the source of a saved odometry run.

`--max-features` defaults to 1500. SIFT can exceed that cap for tied responses.
`--pair-offsets` defaults to `1 2 5 10 20`, measured in positions in the selected
view list. With stride 2, these normally mean original frame gaps 2, 4, 10, 20
and 40. Gaps and original frame indices are exported explicitly. Controls compare
the same selected views using only offset 1, halve both pixel and metric gates,
and rebuild the baseline to check every saved scientific array exactly.

For fresh default Sparse odometry followed by tracks and controls:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python \
  experiments/multiview_constraints.py \
  --tum-path data/tum/rgbd_dataset_freiburg1_rpy \
  --start 60 --stop 200 --stride 2 --output-dir outputs/multiview/rpy
```

Replace `--tum-path` with `--sens-path` for ScanNet. Ranges use the existing
adapter's associated frame numbering. The odometry reader resets frame indices
within each run. Add the saved source start to recover the original source index.
Inputs, dependency versions, package source hash, configuration, trajectory hashes
and selected-frame fingerprints are retained in the reports. Use fresh outputs
for repeats. Timings and compressed archive bytes need not reproduce exactly.

## Architecture and identity

- `scene_recall.features` owns reusable SIFT extraction and conservative RGB-D
  sampling. Sparse VO calls the same operations as before, retaining its cache,
  matching policy, pose fitting, defaults and quality checks.
- `multiview.model` defines `FeatureFrame`, `TrackConfig`, `Landmark` and
  `TrackResult`. A measurement key is `(frame_index, feature_row)`. Extraction is
  done once per selected observation, so all tested pairs share the same rows.
- `multiview.tracking` builds candidate edges and merges compatible components.
  `build_tracks` accepts explicit frame-index pairs for known nonlocal revisits.
  The default schedule is bounded temporal matching, without image retrieval.
- `multiview.geometry` constructs depth-anchored landmarks and validates every
  observation. `multiview.diagnostics` measures support, covisibility and residuals.
- `multiview.runner` consumes saved odometry through verified dataset replay.
  `multiview.io` publishes JSON and flat numeric tables atomically and provides
  `load_measurements` for consumers that do not need descriptors or images.

The public API accepts canonical calibration and reference-free `FeatureFrame`
objects. `FeatureFrame.from_trajectory` copies the estimated coordinate convention
as `T_SR = T_SC_D @ inverse(T_RD)`. Feature depth samples are in RGB optical frame
R, and landmarks are in segment frame S, in meters. Reprojection uses `K_R` and
`inverse(T_SR)`. RGB and depth grids can differ. Nonidentity extrinsics use the
same visibility-aware nearest-surface registration as Sparse VO. Missing depth
remains NaN, and rejected foreground cannot expose background samples.

Track IDs identify final components within one result. They are deterministic
for identical extracted inputs and candidate pairs, assigned by the smallest
measurement key. They are not a cross-run or online identity service. Changing
the range, feature budget, pair schedule or extraction can change the IDs.
Unmatched features remain available as measurements without singleton landmarks.

## Association and landmark policy

1. Require reciprocal SIFT matches with Lowe ratio below 0.7 in both directions.
   Both directions need two neighbors. Descriptor ties and ambiguous repeated
   descriptors fail. Each pair has one-to-one feature assignments.
2. Use each available endpoint depth to project into the other estimated camera.
   Require positive camera depth and symmetric reprojection error at most 3 px.
   If both depths exist, require their transformed 3D discrepancy at most
   `0.03 m + 0.01 * min(z_source, z_target)`. At least one endpoint needs depth.
3. Sort proposals deterministically by descriptor ratio, geometric score and
   measurement keys. A component may contain only one feature from each view
   and only one trajectory segment. Duplicate-view conflicts are rejected.
4. For each proposed merge, transform all valid depth samples into S and take
   their coordinate median. Require that point to satisfy every member's 3 px
   reprojection gate and every depth member's `0.03 m + 0.01 * z` 3D gate.
   There is no silent outlier pruning. This prevents gradual transitive drift
   from building a long inconsistent component from locally acceptable edges.
5. Mark a landmark `supported` with at least three views and two depth samples.
   Smaller or single-depth components are `tentative`. Both pass strict geometry
   under the supplied poses. These labels describe measured support, without
   assigning probabilistic confidence or claiming externally verified identity.

Missing-depth views can join through a depth-supported endpoint and contribute
pixel measurements. Two missing-depth endpoints cannot establish a direct edge.
Depth-free triangulation, delayed monocular initialization and descriptor-only
identity merges are deferred. Multiple accepted edges within a component are
retained as direct cycle checks.

The median limits sample influence but does not model pose or sensor covariance.
It is coordinate dependent and is an initialization rather than a statistically
optimal landmark estimator. Robust acceptance comes from checking all members.
An incompatible merge leaves its existing components intact. The greedy order
does not revisit a rejected merge after subsequent component changes, so it can
underestimate usable support.

## Uncertainty and diagnostics

Estimated poses are treated as fixed hypotheses. Match residuals between the
strict gate and twice that gate are `uncertain` audit records and never merge
identities. Larger errors, absent depth, conflicts and incompatible components
have separate statuses or reasons. A correct correspondence can be rejected
because of VO drift, calibration approximation, RGB/depth timing skew or noisy
depth. These causes cannot be separated solely from these residuals. Gate
thresholds are tolerances, without a claim of calibrated uncertainty.

Reports retain lengths, frame/time spans, depth support, RGB grid occupancy,
metric baselines, viewing-ray parallax and residual distributions. The 6 by 8
occupancy grid avoids interpreting a large bounding box around a few points as
broad support. Per-view records include feature availability and retained support.

Covisibility counts shared landmarks for every pair of views within a component,
even when that pair was not directly matched. Direct candidate edges and inferred
covisibility remain separate tables. Supported covisibility includes the minimum
grid coverage across the two views and the estimated camera baseline. A graph
edge counts identities, without implying independent measurements or full-rank
pose information. Low-parallax or pure-rotation tracks remain identifiable and
are reported rather than being promoted as strong translation constraints.

Leave-one-depth-out diagnostics reconstruct each tested view from the median of
the other depth samples. The report includes pixel and metric errors, strict pass
fraction and invalid projections. This tests dependence on an individual depth
sample. Association gates and VO already used related measurements, so it is not
an independent external validation set. No GT-based accuracy claim follows from
small residuals.

Geometrically compatible repeated textures and coherent moving objects can still
produce incorrect identities. Multiple SIFT orientations at the same location
can also yield separate, correlated tracks. Counts therefore overstate the
number of statistically independent physical landmarks. These experiments do
not estimate real-data identity precision against annotated ground truth.

The adjacent-only control tests whether skip-view matching changes support.
Additional counts identify tracks with missing selected views, tracks whose
members are not connected by adjacent direct edges, and direct nonconsecutive
associations. A lower number of longer tracks can be preferable to a larger
number of short fragments. Compare observation count, extent and covisibility
together with track count.

## Saved measurement contract

`measurements.npz` is pickle-free format 1. `frame_index`, `segment_id`,
`timestamp_s` and `T_SR` describe the selected cameras. `feature_offsets` indexes
the concatenated `pixels` and `points_R` tables. `track_offsets` indexes
`observation_keys`, per-member pixel errors and metric depth errors. Each key
references one frame and one local feature row. Track tables include metric
`point_S`, status, segment, depth support, baseline and parallax. Edge tables
include endpoints, ratio, residuals, status and reason. Candidate frame pairs are
saved even when no descriptor survives. `report.json` supplies calibration,
thresholds, source verification, environment and diagnostic populations.

`load_measurements(run)` returns the report and numeric tables, checking offsets,
feature references, duplicate membership, per-view uniqueness, segment ownership,
canonical measurements, poses and basic report agreement. Descriptors and RGB
images are not embedded. Re-association requires replay and extraction. Consumers
must fix a pose gauge separately in each segment and retain original observations
rather than treating every pairwise covisibility edge as new independent data.

## Validation and limits

Tests use controlled local geometry, deterministic descriptors and synthetic
RGB-D images. They exercise exact nontrivial motion, nonidentity extrinsics,
unequal grids, wrong matches, ambiguous descriptors, missing depth, nonconsecutive
recovery, behind-camera projections, pose-sensitive rejection, transitive drift,
duplicate-view conflicts, segment isolation, reference independence, archive
integrity, source verification and failed publication.

This remains a bounded offline frontend. Feature extraction, descriptor matching,
track construction and diagnostics are timed separately. The runner retains
selected descriptors and all tested edges until export. Covisibility and final
extent calculations can be quadratic in a track's number of views. Use bounded
ranges or a stride for sequence investigations. There is no streaming eviction,
dynamic-object model, global place recognition, track splitting with reassignment,
calibrated noise model or trajectory optimization.

See [the measured investigation](multiview_evaluation.md) for the evaluated
sequences, controls and decision about the next optimization milestone.
