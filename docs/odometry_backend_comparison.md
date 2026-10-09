# RGB-D backend comparison

October 9, 2026. All three backends use the same canonical observations,
sequential tracking, reference evaluation, and export infrastructure. The sparse
estimator remains the established default. Hybrid runs its complete accepted
SIFT, matching, PnP/RANSAC, and LM estimate before depth refinement. Independent
ICP initializes from identity and uses no RGB information or sparse estimates.

## Method

The controlled matrix covers `scene0000_00` ranges `[0, 200)`, `[1000, 1200)`,
`[2000, 2200)`, `[3500, 3700)`, and `[4500, 4700)`. Each backend sees 1000 frames
and 995 attempted transitions across the five independent clips. These ranges
reuse earlier experimental portions, including later motion and the known
ten-frame reference discrepancy. Only this capture is installed locally.
Results across several captures remain unmeasured.

Default settings are held fixed. Standalone sparse and ICP settings match their
hybrid components exactly. The runner executes serially with one OpenCV thread,
one OpenBLAS thread, and one OMP thread. Backend order rotates across ranges.
Each run reopens and decodes the same capture. Reported tracking time includes
opening, indexing, image decoding, and estimation, excluding evaluation and
export. There is one timing measurement per matrix entry. Cache, system load,
and hardware can still influence those descriptive runtime measurements.

Inputs, source indices, untouched reference arrays, calibration, component
settings, software versions, thread settings, and package fingerprints are
checked for consistency before comparison. RPE at intervals 1, 5, and 10 uses
identical reference-valid pairs whose endpoints share a segment in every
backend. Each run also reports its own accepted population. Accuracy alone on
short segments can conceal tracking losses, so coverage, segment count, and
largest segment accompany the pose errors.

Drift uses segment anchoring at the first valid reference, without further
fitting. Aligned ATE additionally fits a proper position alignment with fixed
metric scale. These measure different aspects of the estimated trajectory.
A full-clip endpoint is available only when tracking stayed in one segment.
Fragmented results are reported by segment and cannot be compared as though
they represented a continuous full-clip trajectory.

Hybrid refinement comparisons use the complete accepted sparse initialization
and successful geometric refinements on identical pairs. Fallback poses are
excluded from the improvement population. Held-out depth scores use identical
source point IDs for the sparse and ICP poses, with associations recomputed
separately. Alternating source tiles are held out from fitting, while target
normal neighborhoods can overlap them. These scores assess the depth objective
and do not provide independent sensor validation.

The acceptance policy is fixed before this comparison and is not tuned against
reference poses. Full local rank, support, finest-scale convergence, and hybrid
correction bounds are numerical and heuristic quality gates. A small accepted
residual does not establish physical pose accuracy.

## Results

Sparse and hybrid tracked all 995 attempted transitions. Depth-only ICP tracked
987/995 (99.20%), with eight finest-scale iteration-limit failures. Hybrid
retained its accepted sparse estimates on eight ICP iteration-limit failures.
All 995 hybrid sparse initializations match the standalone sparse poses exactly.
All historical sparse archive arrays match the fresh runs in every tested range.

The aggregate RPE populations contain 987, 937, and 886 identical pairs at
intervals 1, 5, and 10. RMSE aggregates weight each clip by its pair count.
Throughput is total processed frames divided by total tracking time.

| Backend | Common RPE 1 mm / deg | Common RPE 5 mm / deg | Common RPE 10 mm / deg | Overall frames/s |
| --- | ---: | ---: | ---: | ---: |
| sparse | 8.621 / 0.222 | 17.600 / 0.442 | 23.721 / 0.580 | 3.43 |
| icp | 9.841 / 0.253 | 31.061 / 0.869 | 52.799 / 1.574 | 2.09 |
| hybrid | 9.298 / 0.253 | 29.766 / 0.866 | 51.163 / 1.568 | 1.33 |

The following table uses common-population RPE 10. Anchored and aligned
translation RMSE and endpoints use the complete clip for continuous runs.
A dash denotes an unavailable full-clip metric after a tracking loss. Full
segment metrics remain in the report. Rotation endpoints are in degrees.

| Range | Backend | Common RPE 10 mm / deg | Anchored RMSE mm | Aligned ATE mm | Endpoint mm / deg | Frames/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [0, 200) | sparse | 25.388 / 0.615 | 69.564 | 33.143 | 41.025 / 1.702 | 3.49 |
| [0, 200) | icp | 48.458 / 1.383 | 138.829 | 75.479 | 297.340 / 5.552 | 2.15 |
| [0, 200) | hybrid | 47.470 / 1.380 | 149.906 | 79.999 | 326.010 / 5.535 | 1.42 |
| [1000, 1200) | sparse | 28.279 / 0.687 | 54.157 | 38.388 | 89.561 / 2.356 | 3.57 |
| [1000, 1200) | icp | 47.089 / 1.675 | — | — | — | 2.01 |
| [1000, 1200) | hybrid | 42.612 / 1.675 | 118.611 | 83.357 | 227.490 / 9.740 | 1.33 |
| [2000, 2200) | sparse | 16.690 / 0.308 | 73.488 | 41.711 | 130.222 / 1.355 | 2.70 |
| [2000, 2200) | icp | 74.076 / 1.939 | — | — | — | 2.07 |
| [2000, 2200) | hybrid | 73.840 / 1.913 | 746.824 | 280.442 | 1157.441 / 23.399 | 1.23 |
| [3500, 3700) | sparse | 21.941 / 0.428 | 105.299 | 49.437 | 180.370 / 1.708 | 3.87 |
| [3500, 3700) | icp | 62.430 / 1.779 | — | — | — | 2.07 |
| [3500, 3700) | hybrid | 62.471 / 1.780 | 535.022 | 209.274 | 732.683 / 18.680 | 1.30 |
| [4500, 4700) | sparse | 23.822 / 0.710 | 75.783 | 38.952 | 47.733 / 1.088 | 3.79 |
| [4500, 4700) | icp | 22.730 / 1.023 | 71.247 | 41.685 | 70.685 / 4.193 | 2.18 |
| [4500, 4700) | hybrid | 14.952 / 1.025 | 57.765 | 31.334 | 71.474 / 4.303 | 1.39 |

![Matched backend accuracy, drift, and throughput](/home/rlyeh/repos/scene-recall/outputs/odometry/backend_comparison/backend_comparison.png)

Sparse remains the strongest default on this matrix. Hybrid's translation RPE
and aligned ATE improve substantially in `[4500, 4700)`. Its RPE 1 translation
falls from 7.204 to 3.971 mm and aligned ATE from 38.952 to 31.334 mm there.
Ten-frame rotation error still rises from 0.710 to 1.025 degrees, and endpoint
translation rises from 47.733 to 71.474 mm. Hybrid worsens ten-frame translation
in the other four clips and endpoint translation and rotation in all five.
No single accuracy metric captures these trade-offs.

Depth-only ICP offers an independent estimator for textureless observations,
but its identity initialization, geometric objective, and convergence gate do
not provide universally better motion. Its eight losses are at absolute targets
1079, 2007, 2016, 2019, 2118, 3548, 3554, and 3563. Largest segment lengths
are respectively 200, 121, 99, 137, and 200 frames across the five clips.
Hybrid's fallback policy preserves sparse continuity but does not rescue failed
sparse estimation. Neither accepted local support nor rank six prevents drift.
The 23.399-degree hybrid endpoint rotation in `[2000, 2200)` is an example of
substantial accumulated error despite almost complete geometric convergence.

## Does geometric refinement improve sparse motion?

All 987 successful refinements reduce common held-out depth RMSE. Translation
reference error improves on 503/987 pairs (50.96%), while rotation improves on
397/987 (40.22%). The same residual reduction accompanies worse translation on
484 pairs and worse rotation on 590 pairs. On this exact successfully refined
population, translation RMSE increases from 8.593 to 9.274 mm and rotation RMSE
from 0.221 to 0.252 degrees. Eight fallbacks are excluded from these counts.

For example, 2003→2004 reduces common held-out depth RMSE from 7.865 to
5.914 mm over 18,373 points. Its translation reference error increases from
3.686 to 23.080 mm and rotation from 0.060 to 0.095 degrees. This target is
away from the ten-frame boundary. The measurements do not support attributing
all geometric disagreements to the earlier periodic reference discrepancy.

![Depth residual reduction and reference pose error changes](/home/rlyeh/repos/scene-recall/outputs/odometry/backend_comparison/refinement_tradeoff.png)

The complete sparse initialization helps define a useful local basin, while
ICP optimizes a different measured objective and can move away from an already
good sparse pose. This comparison supports retaining hybrid as an explicit
experimental choice. It does not support enabling depth refinement by default
or accepting a candidate because its residual is lower. Initialization,
normal estimation, measurement bias, correspondence changes, and reference
errors remain possible contributors. Their individual causal roles are not
isolated by this matrix. Further tuning should use additional captures and
independent references, with drift and reliability assessed alongside local RPE.

## Interpretation limits

The reference poses come from reconstruction and are not an independent motion
measurement. Earlier [depth diagnostics](depth_registration_experiment.md)
found evidence of discrepancies at some ten-frame boundaries. Reference-relative
errors remain useful for controlled comparison, while physical pose accuracy and
the cause of those discrepancies remain uncertain. Shared depth input also
prevents agreement with the reconstruction from establishing external validity.

The experiment covers five 200-frame clips in one indoor capture, adjacent-frame
motion, and one fixed calibration. No timestamps are available. Rates are
processing throughput, rather than demonstrated real-time sensor operation.
The matrix does not test large initialization errors, independently measured
poses, moving objects, broad cross-scene generalization, or loop closure.

Finest-scale convergence is intentionally strict. Discrete projective
associations can oscillate at an iteration cap even with substantial support
and rank six. Such candidates remain available for inspection. Relaxing that
gate would change the acceptance population and should be evaluated separately.
Likewise, held-out residuals and forward/backward agreement can diagnose a fit
without proving correct motion. A systematic geometric bias can survive all
local convergence checks and accumulate over many accepted transitions.

## Reliability probe

A separate 20-frame clip replaces RGB at local frames 5–7 with black images and
depth at frames 12–14 with missing measurements. Each outage affects adjacent pairs at its boundaries. The probe keeps calibration, references, and all
other measurements unchanged. It reports exact failure indices and segment
boundaries for each backend. The next usable adjacent pair is attempted
normally. This tests explicit outage policy and does not establish general
robustness to blur, motion, moving objects, or plausible but biased measurements.

```bash
SENS=/home/rlyeh/datasets/scannet/scans/scene0000_00/scene0000_00.sens
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/odometry_failure_probe.py \
  --sens-path "$SENS" \
  --output-dir outputs/odometry/backend_failure_probe_repeat
```

The measured probe records sparse and hybrid losses at local targets
5, 6, 7, 8, 13, 14, and 15. Depth-only ICP keeps tracking through black RGB
and loses targets 12, 13, 14, and 15 on missing depth. Sparse PnP can estimate
11→12 using valid source depth and target RGB. Hybrid retains that sparse pose
when ICP reports insufficient support in target depth. The sparse-based methods
resume at 8→9 and 15→16, while ICP resumes at 15→16. Every loss starts an
independent segment at identity, and evaluation excludes all cross-segment RPE.
The probe report is `outputs/odometry/backend_failure_probe/report.json`.

## Verification

All 422 tests pass. The new tests cover analytic motion and transform direction,
RGB and reference independence, missing depth, planar degeneracy, convergence
caps, strict and fallback policies, correction bounds, stage export, shared
comparison populations, and misleading residual improvements. Existing
finite-difference and analytic-room registration tests now exercise the shared
geometry primitives. Lint, formatting, and whitespace checks pass.

All three backends were repeated on `[0, 20)`. Their complete archive arrays
match the corresponding matrix prefixes exactly. Numerical diagnostics match
after excluding stage runtime. The package fingerprints match across all matrix
runs and repeats. The sparse matrix also exactly reproduces every existing
historical archive array across all five clips.

## Reproduction

```bash
SENS=/home/rlyeh/datasets/scannet/scans/scene0000_00/scene0000_00.sens
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/scene-recall-uv-cache \
uv run --offline python experiments/compare_odometry_backends.py \
  --sens-path "$SENS" \
  --output-dir outputs/odometry/backend_comparison_repeat
```

The measured matrix is in `outputs/odometry/backend_comparison/`. Each backend
and range has a `report.json`, `trajectory.npz`, and `frames.csv`.
`comparison.json` records matched errors, refinement outcomes, failure reasons,
configuration, timing, and report hashes. Generated artifacts remain ignored
by Git. Existing outputs are preserved. Use a fresh directory when rerunning.

The architecture, configuration, coordinate conventions, failure policies, and
version 2 export schema are documented in [odometry.md](odometry.md).

Optional figures use Matplotlib 3.11.2 without changing project dependencies:

```bash
MPLCONFIGDIR=/tmp/scene-recall-mpl uv run --with matplotlib==3.11.2 python \
  experiments/plot_odometry_backends.py outputs/odometry/backend_comparison_repeat
```

PNG and SVG figures were inspected visually. No commit was created.
