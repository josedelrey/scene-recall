# Initial sparse odometry evaluation

Experiments used the local ScanNet `scene0000_00` capture on October 9, 2026.
It contains 5578 records. Tested ranges had valid reference poses and unavailable
color and depth timestamps. Intervals below refer to frames. These results
characterize one capture, rather than general performance across datasets.

Run instructions and the output schema are in [odometry.md](odometry.md).
Full distributions, diagnostics, configuration, calibration, input identity, and
software versions are in `outputs/odometry/<run>/report.json`. Trajectories and
frame diagnostics are saved alongside each report. Generated runs are ignored
by Git.

## Controlled 200-frame ablations

All runs processed `[0, 200)`, tracked all 199 transitions, and produced one
segment. Settings differ only in the named option. Numbers are RMSE unless
labeled endpoint.

| Run | RPE 1 translation mm | RPE 1 rotation deg | RPE 10 translation mm | RPE 10 rotation deg | Aligned ATE mm | Endpoint translation mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default_0_200` | 9.302 | 0.247 | 25.388 | 0.615 | 33.143 | 41.025 |
| `no_lm_0_200` | 11.640 | 0.293 | 35.742 | 0.849 | 43.945 | 34.835 |
| `no_edges_0_200` | 9.190 | 0.243 | 25.855 | 0.618 | 29.760 | 33.898 |
| `no_mutual_0_200` | 9.276 | 0.246 | 25.656 | 0.616 | 28.898 | 25.909 |
| `low_ransac_0_200` | 9.304 | 0.247 | 25.648 | 0.620 | 31.017 | 41.905 |

LM reduced one-frame translation RMSE by about 20% and ten-frame translation
RMSE by about 29%. It improved aligned ATE. Accumulated drift did not uniformly
improve. Anchored translation RMSE increased from 58.896 mm without LM to
69.564 mm with LM. Endpoint rotation error increased from 1.064 degrees to
1.702 degrees. Local, aligned, anchored, and endpoint metrics answer different
questions and should be considered together.

Depth-edge and mutual-match filtering had small, mixed effects. Removing either
improved aligned ATE while slightly worsening ten-frame RPE. Both remain
configurable. These results do not establish general accuracy improvements from
either filter.

The higher RANSAC budget had little effect on these consecutive pairs. It helped
the earlier isolated `0 -> 10` pair. Consecutive tracking and wider-baseline
matching have different outlier regimes. Keep the budget configurable and retain
the pair experiment for stress testing.

## Longer and later ranges

| Default configuration range | Tracked transitions | RPE 1 mm / deg | RPE 10 mm / deg | Aligned ATE mm | Endpoint mm / deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[0, 1000)` | 999 / 999 | 8.362 / 0.205 | 23.614 / 0.547 | 69.104 | 122.976 / 2.076 |
| `[3500, 3700)` | 199 / 199 | 9.155 / 0.202 | 21.952 / 0.430 | 49.437 | 180.370 / 1.708 |

Both runs produced one segment. Good adjacent-frame accuracy still accumulated
substantial drift. The later range also checks source-index rebasing and
anchoring away from source frame zero.

The 200-frame default configuration was rerun into `default_0_200_repeat`.
Every archived array and the complete evaluation matched the first run exactly.
The rerun included registration safeguards for separate sensor frames that
leave ScanNet's shared-frame path unchanged. Fingerprints record that code
difference. Synthetic same-code repetitions also test deterministic export.

The historical `0 -> 10` experiment was rerun into a temporary directory.
Correspondences, pose, inlier indices, errors, and reference transform matched
the pre-existing archive exactly. Historical outputs were not overwritten.

## Limits and next experiments

No natural tracking failures occurred in these ranges. Synthetic tests cover
blank images, failed refinement, unsupported estimates, independent segment
origins, export of lost transitions, and exclusion of cross-segment errors.
Zero detected failures on this capture does not establish reliable detection
under blur, moving objects, repeated texture, or weak geometry.

The fitting objective uses source depth and image correspondences. Low
reprojection error can favor biased observations or compensate rotation with
translation. The frame-10 image discrepancy remains unresolved. Thresholds and
coverage are heuristic quality checks, rather than calibrated uncertainty.

Prioritize multiple captures and an independent reference benchmark before
tuning defaults. Compare optical flow and geometric refinement on the same
ranges and metrics. Keyframes or a local map are the next architectural step for
limiting accumulated drift. LM's local improvements support retaining it as the
initial baseline, while these drift measurements show the need for temporal
and geometric constraints.

Timing includes decode and feature extraction. Some ablations ran concurrently,
so their wall times do not support comparative performance claims. Measure
runtime serially under a fixed environment when evaluating performance.
