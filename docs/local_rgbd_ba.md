# Experimental local RGB-D bundle adjustment

This opt-in offline backend tests shared landmark refinement against the
[matched temporal fixed-landmark optimizer](temporal_fixed_landmarks.md).
Sparse remains the default. The quantitative evaluation and promotion decision
are recorded in [the experiment report](local_rgbd_ba_evaluation.md).

## Matched mathematical model

For RGB camera-to-window pose `T_i`, shared window point `P_l` and frozen
native-depth-to-RGB transform `M_i`, use exactly the saved direct factors:

\[
q_R=T_i^{-1}P_l,\qquad q_D=M_i^{-1}q_R,
\]

\[
E(T,P)=\frac12\sum_{i,l}w_l\left[
\rho(\|\pi_R(q_R)-u_{il}\|^2)
+m_{il}\rho(\|W_{D,il}(q_D-p_{D,il})\|^2)\right].
\]

RGB and native 3D depth have separate Cauchy blocks with scale 2.5.
RGB sigma is one pixel. Native covariance retains the engineering depth sigma
`0.005 + 0.003 z_D²` meters, native sampling uncertainty and frozen motion
allowances. Observation weights sum to one per track for RGB, with the same
per-observation depth weights. Missing depth contributes zero information.
Nothing estimates covariance, clocks, motion bridges, calibration or associations
during optimization. References and held-out RGB/depth never enter this API.

The only additional variables are three coordinates per selected track. Initial
points remain the original window geometric medians of Sparse-transformed native
measurements. There is no point prior. Each shared point has one identity within
the window, including observations in fixed cameras. Points are reinitialized
from the original packet in the next window. They are not carried as new global
map variables. Reconstruction continues to fuse original training depth using
the resulting camera poses, rather than substituting optimized feature points.

The first camera and every committed overlap camera are fixed exactly. This
removes the rigid gauge independently for every trajectory segment. Metric depth
supplies scale conditionally on its calibration. A coherent scale error remains
unidentifiable from these observations. Root fixing cannot recover true scale.

## Numerical method and safeguards

Analytic pose and point derivatives produce IRLS normal blocks `Hpp`, `Hpl` and
independent 3×3 `Hll` blocks. Diagonal LM damping is applied to both variable
families before Schur elimination:

\[
S_\lambda=H_{pp}+\lambda D_p
-\sum_l H_{pl}(H_{ll}+\lambda D_l)^{-1}H_{lp},
\]

\[
S_\lambda\delta_p=-g_p+
\sum_l H_{pl}(H_{ll}+\lambda D_l)^{-1}g_l.
\]

Back-substitution recovers point updates. Only the small camera system is solved
densely. This is the standard block elimination described in the
[Ceres solver documentation](https://github.com/ceres-solver/ceres-solver/blob/master/docs/source/nnls_solving.rst).
The implementation uses NumPy and the existing SE(3) retraction, with no new
runtime dependency. It does not construct a dense joint Hessian in production.

The exact existing Sparse Schur gate determines the common window population.
The joint solver additionally checks undamped point rank and normalized camera
Schur conditioning at each iteration. An invalid point block rejects the window
rather than damping away unobservability or silently deleting measurements.
A later condition failure rolls back the entire window. These local curvature
checks do not establish calibrated uncertainty or correct measurement identity.

Accepted steps must decrease the unchanged robust objective. Cheirality failures
reject proposals without dropping factors. Camera corrections retain the fixed
optimizer's 10 cm / 5 degree bounds and 20-iteration budget. A separate 10 cm
maximum displacement from each point initializer bounds this first experiment.
This is an explicit numerical safeguard, without an independent geometric prior
or a claim that 10 cm is physically optimal. Reports retain bound rejections,
damping, termination reasons, point movement, conditioning and accepted costs.

All committed camera poses and each closed window's final optimized geometry are
retained for a final objective check. No subsequent window can change them.
Independent losses remain independent. No online tracking, loop closure,
marginalization or global objective is introduced.

## Controlled comparison and reproduction

Generate fixed-landmark packets if unavailable, using the original Sparse run:

```bash
uv run --locked python experiments/temporal_fixed_landmarks.py \
  --odometry-run PATH_TO_SPARSE_RUN \
  --output-dir outputs/joint_ba/controls/xyz

uv run --locked python experiments/local_rgbd_ba.py \
  --fixed-run outputs/joint_ba/controls/xyz \
  --output-dir outputs/joint_ba/primary/xyz \
  --evaluate-maps --map-chunks 0 6
```

Use the full TUM XYZ, Desk and RPY streams with map chunks `0 6`, `0 4` and
`0 5`. Preserve the existing offset-two held-out split and mapping settings.
`--maps-only` evaluates a saved result. `--map-output-dir` keeps a separately
labelled diagnostic map population outside the primary report.

The runner verifies input hashes and re-solves every supported fixed packet,
requiring exactly its saved numerical diagnostics. It copies clock-only and
fixed results unchanged into the fresh output. Every supported packet also gets
an isolated joint fit from exactly the saved fixed camera initialization.
These paired costs isolate added point freedom. Sequential joint fitting uses
the same points, factors, masks, gates, free cameras and Sparse initialization,
but later camera starts inherit earlier joint corrections. That unavoidable
initialization difference is recorded per window. Paired window costs are
repeated-observation diagnostics, not independent statistical samples.

The default rejects unknown acquisition clocks. ScanNet can run only with
explicit assumed-static source packets and `--allow-unknown-static` on the joint
runner. Such a result is labelled as an assumption. It cannot validate temporal
compensation. TUM timestamps remain acquisition proxies with unresolved exposure
midpoints, latency and rolling shutter. Frozen motion also inherits Sparse's
mixed-time translation approximation.

```bash
uv run --locked python experiments/joint_ba_controls.py \
  --output outputs/joint_ba/synthetic.json
uv run --locked python experiments/replay_joint_packets.py \
  outputs/joint_ba/primary/xyz outputs/joint_ba/primary/desk \
  outputs/joint_ba/primary/rpy --output outputs/joint_ba/replay.json
uv run --locked python experiments/summarize_joint_ba.py \
  outputs/joint_ba/primary/xyz outputs/joint_ba/primary/desk \
  outputs/joint_ba/primary/rpy --output outputs/joint_ba/summary.json
```

Each experiment repeats poses, point arrays and diagnostics exactly before
publication. Independent replay starts again from the parent packets and checks
all joint arrays and full-rate completed poses. Packet replay does not reopen
source media. Raw preparation can separately be verified using
`experiments/replay_temporal_packets.py --verify-preparation` on the fixed run.
The inherited full-rate Sparse initializer previously consumed the original
stream. The additional optimization uses training observations only. Reports
distinguish shared preparation, fixed and joint numerical solving,
paired diagnostic solves and mapping. Output destinations must be fresh.

Voxel and TSDF evaluation uses the same native training depth, original source
verification and held-out populations. Reference-target scores are primary.
Estimated-target scores remain diagnostic. Report missing-penalized error,
coverage and exact fixed/joint common-pixel comparisons together with pose
translation, orientation, endpoint errors and longer-horizon RPE. Lower fitting
cost alone is insufficient evidence for further development.
