# Multiview constraint investigation

Measured on 2026-10-19 on Windows with Python 3.12.13, NumPy 2.5.3 and OpenCV
5.0.0. These are focused clips, covering 680 RGB-D observations and 311 selected
feature views. They include three Freiburg 1 TUM sequences and two ranges of
one ScanNet capture, without a claim of independent-scene generalization.

The results justify a subsequent small-window pose-only experiment followed by
local RGB-D Bundle Adjustment. They do not establish that optimization will
improve trajectories or held-out map geometry. No optimizer is implemented here.

The subsequent [pose-only experiment](pose_only_multiview.md) and
[its evaluation](pose_only_evaluation.md) now test that recommendation while
preserving this track-only milestone's protocol and results.

## Protocol and retained runs

Default Sparse VO supplies fixed estimated poses. Multiview settings are fixed
before these comparisons: 1500 SIFT features, reciprocal bidirectional ratio 0.7,
3 px reprojection, metric gate `0.03 m + 0.01 * z`, 5 cm depth-edge rejection,
selected-view offsets 1/2/5/10/20, and a twice-threshold audit-only uncertainty band.
All references stay outside association and landmark estimation. The full saved
source is verified before any multiview output is published.

| Clip | Original adapter range | Feature stride | Selected views | VO segments | Retained tracks directory under `outputs/multiview/final/` |
| --- | --- | ---: | ---: | ---: | --- |
| TUM XYZ | `[0, 120)` | 2 | 60 | 1 | `xyz_repeat/` |
| TUM Desk | `[0, 180)` | 3 | 60 | 1 | `desk/tracks/` |
| TUM RPY | `[60, 200)` | 2 | 71 | 2 | `rpy/tracks/` |
| ScanNet start | `scene0000_00 [0, 120)` | 2 | 60 | 1 | `scannet_start/tracks/` |
| ScanNet later motion | `scene0000_00 [3500, 3620)` | 2 | 60 | 1 | `scannet_motion/tracks/` |

RPY's extra selected view is its segment root. Its loss occurs at original
source frame 115, local frame 55, with insufficient final PnP inliers. No identity
or landmark crosses this boundary. ScanNet and the other TUM clips have no VO
losses. The RPY clip exercises an actual failure rather than an injected outage.

Use `experiments/multiview_constraints.py` with the table's range and stride to
reproduce each case in a fresh directory. TUM sources are
`data/tum/rgbd_dataset_freiburg1_{xyz,desk,rpy}`, and ScanNet is
`data/scannet/scans/scene0000_00/scene0000_00.sens`. See the
[measurement guide](multiview_tracks.md#run-and-reproduce) for commands.
Desk, RPY and ScanNet odometry are retained beside their tracks. XYZ uses
`outputs/multiview/xyz120/odometry/` and a fresh independent replay into
`xyz_repeat/`. Generated outputs are ignored, without embedding source images
or dataset files in Git.

## Track support and extent

Lengths count selected observations. `supported` requires at least three views,
two depth samples and strict all-member validation. Remaining tracks are
tentative. Counts do not represent statistically independent physical points.

| Clip | All tracks | Supported | Track observations | Length median / p90 / max | Span p90 / max in original frames |
| --- | ---: | ---: | ---: | --- | --- |
| XYZ | 10,490 | 5,794 | 51,887 | 3 / 11 / 55 | 40 / 118 |
| Desk | 7,341 | 3,367 | 24,729 | 2 / 6 / 24 | 21 / 87 |
| RPY | 4,940 | 2,573 | 18,967 | 3 / 7 / 31 | 20 / 76 |
| ScanNet start | 3,087 | 1,654 | 13,650 | 3 / 9 / 50 | 32 / 118 |
| ScanNet later motion | 4,590 | 2,402 | 20,540 | 3 / 9 / 37 | 28 / 94 |

TUM p90/max time spans are XYZ 1.403/4.138 s, Desk 0.737/3.308 s and RPY
0.668/2.538 s. These ScanNet clips provide no usable canonical timestamps, so
their extents are reported in frames without assuming a frame rate.

After conservative sampling, valid depth is available at 75.2–81.2% of all
extracted feature rows. Among retained track observations it rises to 86.1–93.6%.
The remaining observations retain their pixel constraints with missing depth.
This reflects conservative selection, without filling or inventing depth.

## Geometric consistency

| Clip | Track observations with depth | Median reprojection, px | Median 3D sample error, mm | Leave-one-depth-out strict pass | Leave-one-depth-out p90, px |
| --- | ---: | ---: | ---: | ---: | ---: |
| XYZ | 86.1% | 0.61 | 5.13 | 98.2% | 1.98 |
| Desk | 92.8% | 0.63 | 3.78 | 98.0% | 2.22 |
| RPY | 88.9% | 0.63 | 3.42 | 98.7% | 2.08 |
| ScanNet start | 91.8% | 0.94 | 5.45 | 97.6% | 2.52 |
| ScanNet later motion | 93.6% | 0.83 | 5.95 | 97.9% | 2.34 |

All retained members pass the fitted-landmark gates by construction. The
leave-one-depth-out check instead predicts a view using the other depth samples.
It has zero invalid projections in these runs, but 1.3–2.4% of tested observations
fail strict gates. It remains conditional on association selection and estimated
poses. Small errors do not measure real-data identity precision, external pose
accuracy or independent map quality.

Between 9.8% and 16.7% of descriptor proposals fall in the relaxed geometry band
and never merge identities. Full-component validation additionally rejects
6,670 XYZ, 1,125 Desk, 1,300 RPY, 1,872 ScanNet-start and 1,614 ScanNet-later
merges. Duplicate-view conflicts reject another 277/29/56/71/48 proposals.
These are incompatible hypotheses under the current poses, without a claim that
every rejected descriptor match is physically wrong.

## Information beyond adjacent association

The adjacent-only control uses offset 1 in the same selected-view list. It is
not a comparison with running VO on temporally subsampled images. VO continues
to consume every source observation in all cases.

| Clip | Adjacent-only observations | Full schedule observations | Gain | Observations with half-sized gates |
| --- | ---: | ---: | ---: | ---: |
| XYZ | 38,807 | 51,887 | 33.7% | 46,137 |
| Desk | 20,148 | 24,729 | 22.7% | 18,989 |
| RPY | 14,604 | 18,967 | 29.9% | 15,992 |
| ScanNet start | 11,494 | 13,650 | 18.8% | 10,357 |
| ScanNet later motion | 16,592 | 20,540 | 23.8% | 15,678 |

In XYZ, p90 track length rises from 5 to 11 views. Its 6,482 tracks with missing
selected views and 7,017 tracks without an adjacent-direct-edge chain demonstrate
recovery beyond merely recording redundant cycles. Other clips also retain
2,706/2,602/1,356/2,296 gap-spanning tracks for Desk/RPY/ScanNet-start/ScanNet-later.
Halving gates reduces observation support by 11.1–24.1%. XYZ's supported track
count nevertheless rises from 5,794 to 6,260 because of fragmentation, showing
why track count alone is an inadequate quality metric.

Covisibility below includes zero-support pairs in each within-segment cohort.
Coverage is the occupied fraction of the fixed 6 by 8 image grid, taking the
minimum across both views. A shared landmark can induce covisibility without
a direct descriptor edge between that particular pair.

| Clip | Gap 10 median supported / coverage | Gap 10 pairs with >=20 supported and >=25% coverage | Gap 20 median supported / coverage |
| --- | --- | --- | --- |
| XYZ | 203 / 44.8% | 49/50 | 74.5 / 32.3% |
| Desk | 10.5 / 12.5% | 15/50 | 0 / 0% |
| RPY | 13 / 8.3% | 8/51 | 1 / 2.1% |
| ScanNet start | 32 / 25.0% | 27/50 | 15.5 / 14.6% |
| ScanNet later motion | 59.5 / 31.3% | 32/50 | 12.5 / 13.5% |

Gaps count selected views, normally 20/40 original frames at gaps 10/20, and
30/60 original frames for Desk. RPY's forced root creates one irregular interval.
The 20-point and 25%-coverage criteria are descriptive cohorts rather than tuned
frontend defaults or guarantees of observable pose parameters. Shorter windows
are much more broadly supported than fixed long windows.

Supported-track median baseline/parallax is XYZ 17.5 cm/2.83 degrees, Desk
19.0 cm/8.17 degrees, RPY 2.43 cm/0.87 degrees, ScanNet-start 8.14 cm/1.55 degrees
and ScanNet-later 21.7 cm/4.11 degrees. Only 42.4% of RPY's supported tracks
exceed 1 degree parallax, compared with 85.1% XYZ, 99.7% Desk, 64.8% ScanNet-start
and 94.2% ScanNet-later. RPY's rotational coverage does not imply equally strong
translation constraints. No normal-matrix rank or conditioning is claimed here.

## Computational cost and repeatability

Final runs set OpenCV, OpenBLAS and OMP to one thread. Timing is descriptive,
without an isolated-host benchmark or cross-sequence speed ranking. Dataset
decoding, image size, feature availability and host load differ.

| Clip | Full verified replay, s | Matching/gating, s | Track construction, s | Diagnostics, s | Baseline phase sum, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| XYZ | 15.0 | 15.5 | 7.9 | 3.6 | 41.9 |
| Desk | 16.0 | 10.5 | 2.8 | 2.0 | 31.3 |
| RPY | 19.0 | 3.9 | 2.6 | 1.3 | 26.7 |
| ScanNet start | 20.5 | 3.3 | 2.3 | 1.0 | 27.0 |
| ScanNet later motion | 20.5 | 5.6 | 2.9 | 1.3 | 30.3 |

Replay includes SIFT extraction, so extraction time must not be added again.
The phase sums exclude VO, controls and archive compression. Controls plus the
baseline take approximately 39–101 s, including final baseline diagnostics but
excluding compression. Track archives occupy 1.7–5.2 MiB without descriptors or
images. Peak RSS is unavailable through the existing Windows resource helper.
This is an offline infrastructure milestone, without a real-time claim.

Each clip's in-memory rebuild matches all 27 scientific archive arrays exactly.
XYZ also matches all 27 after independent source replay and feature extraction,
including matching control summaries. All final archives pass the measurement
loader's integrity checks. A comparison against Sparse VO loaded from Git HEAD
matches estimated poses and diagnostics exactly on 12 real XYZ frames. No
odometry or mapping algorithm, default or saved format changed.

All final multiview runs record package source SHA-256
`9b1e31abdc40d91a8929a7f1bece5fe2dc889d91f2e534743bae93637217a8df`.
Raw reports, the derived `outputs/multiview/summary.json`, and
`outputs/multiview/verification.json` retain detailed populations and evidence.
Exact repeatability is scoped to the recorded software environment.

The 33 new synthetic tests pass. The final full native Windows test run reports
692 passed, 2 skipped for unavailable symbolic-link privilege, and one failure
caused by intermittent `WinError 5` during an existing odometry-directory rename.
That test and all cases affected in earlier attempts pass targeted retries.
The error also occurs outside the sandbox, and its cause remains unresolved.
No numerical or identity assertion fails. Ruff lint, formatting and whitespace
checks pass. Linux and remote CI were not run for this milestone.

## Next milestone and prerequisites

Proceed with small, per-segment pose-only windows on supported tracks before
joint RGB-D BA. XYZ provides the clearest longer-range evidence. ScanNet and
Desk provide useful local constraints, while RPY requires particular attention
to baseline, spatial coverage and conditioning. Select windows using those
properties rather than a fixed length or raw track count.

Before interpreting optimization gains:

- Fix the gauge independently in every segment and consume original per-view
  measurements once. Covisibility edges are derived relationships, without
  introducing extra independent residuals for each pair.
- Establish robust pixel/depth residual weighting and an explicit noise policy.
  Current thresholds are tolerances. Pose covariance, RGB/depth time skew,
  TUM's pinhole approximation and calibration error remain unresolved sources
  of association fragmentation and biased landmark initialization.
- Check window conditioning and correlated SIFT observations. Geometrically
  compatible repeated texture and moving objects can survive these gates.
  Real-data identity precision has not been measured against annotations.
- Test pose-only behavior with the same tracks and evaluate joint BA against
  fixed held-out map evidence. Re-association after a pose change should be a
  separately measured step because the current selection depends on VO poses.
- Address intermittent Windows publication failures before relying on unattended
  repeated test or experiment execution. Avoid drawing performance conclusions
  from the present timings.

Depth-free initialization, global retrieval, cross-segment association, online
track eviction and a full SLAM frontend remain outside this stage. More scenes
and motion regimes are needed before generalizing the measured constraint quality.
