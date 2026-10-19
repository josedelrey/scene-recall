# Experimental temporal RGB-D fixed-landmark control

This offline experiment addresses the measurement and comparison prerequisites
identified by the [BA audit](rgbd_ba_prerequisites.md). Camera poses are the only
optimized variables. Joint landmark optimization, online changes, clock-offset
fitting and continuous-time trajectory estimation remain outside this milestone.

Measured outcomes, costs and remaining restrictions are in the
[evaluation report](temporal_fixed_landmarks_evaluation.md).

## Clock and motion policy

Dataset adapters supply spatial calibration and depth units as before. The
experiment interprets documented source frame timestamps as acquisition-time
proxies. It does not claim they measure exposure midpoints or calibrated sensor
latency. TUM supplies color and depth epoch seconds. ScanNet stores source
microseconds, with zero meaning unavailable in the tested captures. Unknown
clocks never become measured zero skew.

The Sparse pose is treated as an RGB-epoch initialization proxy, with the
canonical `T_RD` conjugation preserved. Its rotation is driven by RGB features,
but translation already depends on mixed-time RGB-D samples. Therefore this is
an explicit approximation, without assuming that Sparse yields exact color
poses. No ground truth determines this interpretation or any clock offset.

For each independent segment, interpolate only its training-view pose proxies
at reported color times. Translation is piecewise linear and rotation follows
the shortest arc. The experiment freezes the within-pair transform

\[
A_i = T_{SR}(t_{R,i})^{-1} T_{SR}(t_{D,i}),\qquad
M_i = A_i T_{RD}.
\]

`M_i` maps native depth optical coordinates at depth time into RGB optical
coordinates at color time. Training intervals longer than 250 ms are unsupported.
One-sided continuation at a segment endpoint is bounded by 20.1 ms. Larger
paired skews are unsupported. Depth measurements are rejected above 3 cm or
2 degrees of estimated within-pair motion. Valid estimated motion may still
complete the output epoch conversion when a depth quality gate rejects samples.
Unsupported motion uses an explicit identity fallback for output completion,
with its reason counted. No interpolation crosses a trajectory loss.

Motion allowances use local estimated speed, neighboring velocity variation,
half the estimated displacement and an assumed 2 ms clock allowance. These are
engineering uncertainty terms, without a calibrated confidence interpretation.
Unobserved acceleration, rolling shutter and systematic timestamp latency remain
possible errors. The bridge is never re-estimated from refined poses.

Default handling of unknown clocks disables depth factors. Tracks without a
usable depth initialization cannot enter this fixed-landmark experiment. An
explicit `--allow-unknown-static` control retains spatial calibration while
assuming coincident exposures. Its label and report identify the assumption.
It does not establish ScanNet synchronization.

The tested TUM archives retain separate frame clocks with median absolute
training-view skew of 5.90, 11.67 and 11.28 ms for XYZ, Desk and RPY. The
[TUM file format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
documents those timestamp fields. Neither it nor the loaded calibration supplies
exposure durations or a calibrated RGB/depth latency. The tested ScanNet source
fields are zero. A frame index or the adapter's synthetic ordering timestamp
cannot substitute for missing acquisition clocks.

Practical alternatives have different contracts:

| Policy | Required assumption and limitation | Additional work |
| --- | --- | --- |
| Static extrinsic only | Pair motion is negligible. Known TUM skew violates this during motion | No temporal preparation |
| Retain only small predicted displacement | Estimated motion bounds exposure displacement. Loses support and still depends on the motion proxy | Motion interpolation and projection, no warp |
| Frozen estimated two-exposure transform, primary | Short local motion interpolation is adequate. Reject uncertain surfaces and retain explicit uncertainty | One native depth projection per training image, covariance and boundary checks |
| Unknown-clock static control | Coincident exposures are assumed. Evidence cannot verify it for these ScanNet clips | Same spatial binding without temporal motion |
| Joint clock or continuous-time motion estimation | Additional temporal observability and calibration would be needed | Additional variables and machinery, deferred |

The gate-only availability diagnostic keeps 51.1%, 34.9% and 22.6% of the
already conservative XYZ, Desk and RPY native bindings below half a predicted RGB
pixel of motion. It was not run as another optimizer and does not measure the
quality of a separately constructed synchronized cohort. The frozen bridge is
the smallest implemented model that uses the available clocks while retaining
explicit bounds and source identities.

## Native depth binding and uncertainty

RGB SIFT extraction uses training images only. Native depth samples are projected
through frozen `M_i` into the RGB grid. A nearest-surface z-buffer retains native
source identities. Search includes adjacent RGB raster cells within a calibrated
native-pixel footprint. Distances are normalized by the focal-length ratios
`fx_R/fx_D` and `fy_R/fy_D`, with a one-native-pixel limit. This matters when RGB
resolution exceeds native depth resolution, as in the tested ScanNet capture.
Requiring the exact same RGB raster cell discards legitimate samples.

Raw foreground wins before depth-edge rejection. Competing candidate surfaces
with more than 5 cm of RGB-frame depth separation are rejected. An invalid
foreground cannot silently expose background. Each native sample contributes
at most once per view, including duplicate SIFT detections. Missing depth stays
missing. Search radius is bounded to 16 RGB cells per axis.

The retained point is the actual native-depth ray sample, with its native source
index and covariance. It is not replaced with a synthetic point on the RGB
keypoint ray. Existing conservative pair matching and all-member validation
rebuild tracks using these corrected camera-frame samples. Nonconsecutive
association policy and independent segment boundaries are preserved.

The native covariance starts with one pixel of grid uncertainty and axial depth
sigma `0.005 + 0.003 z_D²` meters, propagated through native backprojection `B`:

\[
C_D=B\operatorname{diag}(1,1,\sigma_z^2)B^T
+\sigma_t^2 I+\sigma_r^2(\|p_D\|^2 I-p_Dp_D^T),\qquad
W_D=\operatorname{chol}(C_D)^{-1}.
\]

This model approximates sampling and association error. Spatial registration,
neighboring depth correlations, calibration error and shared motion uncertainty
are not independent sensor noise. The covariance is frozen before optimization.
It is not fitted from ground truth or from the gated residual population.

Covariance inflation alone cannot prevent selecting a wrong surface. The
primary policy additionally requires native depth support beyond a motion
footprint of radius `ceil(2 sqrt(trace(J_D C_motion J_D^T)))` native pixels. A
distance transform of the existing 3x3 continuity mask rejects nearby holes and
depth boundaries. The multiplier is an engineering safety margin, without a
probability claim. `--motion-margin 0` disables the additional margin as a
diagnostic control. Associations must be rebuilt after any rejection. Filtering
an already merged track graph would retain evidence from removed samples.

## Matched fixed-landmark objective

Each selected track initializes one point from the geometric median of measured
window depth samples transformed by Sparse. This initializer is equivariant to
rigid changes of segment axes. No saved reference-derived point or initial-point
prior enters the objective. The point is then held fixed.

For a shared point `P_l` and camera `T_i`, define

\[
q_R=T_i^{-1}P_l,\qquad q_D=M_i^{-1}q_R,
\]

\[
E=\frac12\sum_{i,l}w_l\left[
\rho(\|\pi_R(q_R)-u_{il}\|^2)
+m_{il}\rho(\|W_{D,il}(q_D-p_{D,il})\|^2)\right].
\]

Pixel sigma is one, `m` marks available depth, and each track's RGB observation
weights sum to one. Depth blocks use the same per-observation weight. Cauchy
loss scale is 2.5. RGB and depth have separate robust blocks so a corrupted
depth sample need not discard an otherwise useful RGB constraint. Root and
overlap observations remain in the objective, even when their camera poses are
fixed. They will constrain points in a future joint experiment.

This model uses RGB pixels and metric native-depth point observations. It differs
from the old anchor-star formulation and from a registered RGB-ray z residual.
Future joint BA must reuse these exact observations, point identities, initial
points, within-pair motion, whitening, loss, weights and boundary policy. Only
the point parameter freedom should change. Per-window factor archives preserve
all these numeric inputs, their fingerprints, root frame and initial poses.

The pose solver uses analytic left-perturbation derivatives, damped IRLS, a
normalized undamped condition guard of `1e8`, and 10 cm / 5 degree update bounds.
Cheirality failures reject proposals without dropping factors. Default windows
contain six training cameras, advance by four, and retain at most 256 tracks.
Support and spatial-coverage gates match the existing pose-only defaults.

A common observability gate also constructs point information blocks and the
pose Schur complement for this exact RGB/native-depth objective, without
computing any landmark update. It uses the frozen Sparse initialization and
the same boundary cameras a future joint backend must fix. An unanchored
component can be invisible to a fixed-point pose condition check. The common
gate prevents interpreting that artificial fixed-geometry support as observable
joint geometry. Schur conditions are local algebraic diagnostics, without a
calibrated pose covariance interpretation.

## Overlap boundaries and output epochs

The primary policy freezes every already committed overlap camera. Subsequent
windows update only their new cameras. Endpoint corrections carry into the
unseen Sparse tail. Closed window poses and fixed geometry cannot change, and
every previous window objective is checked again after stitching. This avoids
introducing a global objective or marginalization scheme.

A first-camera-only `open` control uses the same factors and initializer but
allows later windows to alter overlapping cameras. Any changes in earlier
objectives are recorded. Locked and open windows also have different subsequent
initializations, so their comparison evaluates the complete boundary policy.

Optimized keyframe corrections are completed over the original full-rate Sparse
poses. Frozen training motion converts RGB-epoch proxy poses back to depth-time
poses for evaluation and native-depth mapping. Each segment is re-rooted to its
first depth camera, exactly identity. Source indices, depth timestamps,
references, losses and held-out partitions are preserved.

The `clock` alternative performs only this epoch conversion. It separates gains
from time interpretation from gains due to fixed-landmark optimization. The
`static` direct-objective control ignores within-pair motion and output epoch
conversion, but shares the primary graph, native bindings and whitening. Its
landmark initializer uses that static transform. Thus it is a conditional
measurement-model control, without claiming an independently constructed static
association baseline. Existing Sparse and anchored/frozen/linear experiments
remain separate alternatives.

## Reproduction and evaluation

```bash
uv run --locked python experiments/temporal_fixed_landmarks.py \
  --odometry-run outputs/pose_only/full_baselines/rpy \
  --output-dir outputs/temporal_fixed/new_rpy \
  --evaluate-maps --map-chunks 0 5

uv run --locked python experiments/temporal_controls.py \
  --output outputs/temporal_fixed/new_controls.json
```

Mapping defaults to Sparse, `clock` and `timed`. To include both other trajectory
controls, pass `--map-variants clock static timed open`. This changes the set of
methods rendered, while preserving training observations and target samples.

```bash
uv run --locked python experiments/replay_temporal_packets.py \
  outputs/temporal_fixed/new_rpy \
  --output outputs/temporal_fixed/new_packet_replay.json
```

Packet replay re-solves the saved windows and checks full-rate pose arrays
exactly, without opening RGB-D media. It checks input and factor fingerprints.
Add `--verify-preparation` to reopen the source and reconstruct all training graph
and native measurement arrays before replay. The default packet-only mode does
not independently reconstruct descriptor association.

Use complete XYZ, Desk and RPY Sparse runs with map chunks `0 6`, `0 4` and
`0 5` respectively. The existing ScanNet-start and later-motion runs cover
`[0,120)` and `[3500,3620)`, with map chunk `0`. Run unknown-clock data first
without assumptions. Use `--allow-unknown-static` in a separate fresh output to
evaluate the explicit assumption. `--maps-only` consumes a completed experiment.
Use `--map-output-dir` for a separate diagnostic map population while preserving
the primary map output. Such populations must be labelled separately.

Raw source replay verifies the canonical fingerprint. Measurement construction
uses only training RGB/depth and training Sparse motion. Full-stream decoding,
including targets, is required for input integrity verification and contributes
to preparation cost. Those decoded target values never enter factors. Full-rate Sparse
initialization is inherited and previously consumed the original stream. Target
timestamps specify output epochs, while their images, depth and references
never construct optimization factors. References are used only after fitting
for external evaluation and reference-target rendering.

All four controls retain the same training observation and target schedules.
Maps use the existing surface-voxel and TSDF evaluation implementation, native
training depth, identical filtering, fixed target samples, missing-coverage
penalties and paired common-pixel metrics. New measurement and factor archive
hashes are verified before mapping. Preparation, factor construction, numerical
solves, completion and mapping costs must remain separate in interpretation.

`training_measurements.npz` stores the training graph with RGB-epoch timestamps.
`native_measurements.npz` retains both reported clocks, native sample IDs,
within-pair transforms, frozen whitening and motion status. Per-method factor
archives retain every window's fixed point initialization, camera initialization
and residual inputs. Canonical trajectory timestamps remain depth timestamps.

Numerical rank, lower residuals and correct synthetic behavior do not establish
better real poses or maps. Evaluate translation, rotation, long-horizon and
endpoint errors together with coverage-aware reconstruction. Report unknown
clocks, temporal assumptions, conditioning failures and regressions explicitly.
