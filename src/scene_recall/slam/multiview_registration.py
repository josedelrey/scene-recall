"""Conditional three-view registration with explicit shared landmark variables.

One source RGB-D sample owns each landmark prior. A historical context supplies
only RGB observations. Incoming fitting tracks share the same latent geometry.
Historical poses are conditioned on, not independent measurements or ground truth.
No conditional Hessian is exported as a graph information matrix.
"""

from dataclasses import replace

import numpy as np

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.geometry.transforms import is_rigid
from scene_recall.multiview.diagnostics import grid_coverage
from scene_recall.multiview.pose_only import projection_jacobian, retract, skew
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.local_map import information_ratio
from scene_recall.slam.metric_refinement import point_covariances
from scene_recall.slam.registration import Registration, register_anchor


def _weighted(
    points, anchors, whitening, context_pixels, context_pose, K, pose=None, pixels=None
):
    """Linearize one source factor and one RGB factor per additional view."""
    context = transform_points(points, context_pose)
    if (context[:, 2] <= 0.05).any():
        raise ValueError("context geometry behind camera")
    prior = np.einsum("nij,nj->ni", whitening, points - anchors)
    residuals = [prior, project_points(context, K) - context_pixels]
    landmark = [
        whitening,
        projection_jacobian(context, K)[:, :2] @ context_pose[:3, :3],
    ]
    camera = [np.zeros((len(points), 3, 6)), np.zeros((len(points), 2, 6))]
    if pose is not None:
        incoming = transform_points(points, pose)
        if (incoming[:, 2] <= 0.05).any():
            raise ValueError("incoming geometry behind camera")
        projection = projection_jacobian(incoming, K)[:, :2]
        residuals.append(project_points(incoming, K) - pixels)
        landmark.append(projection @ pose[:3, :3])
        camera.append(
            projection
            @ np.concatenate(
                (np.broadcast_to(np.eye(3), (len(points), 3, 3)), -skew(incoming)),
                axis=2,
            )
        )
    cost = 0.0
    for i, residual in enumerate(residuals):
        norm = np.linalg.norm(residual, axis=1)
        cost += float(np.where(norm <= 2.5, 0.5 * norm**2, 2.5 * (norm - 1.25)).sum())
        weight = np.sqrt(np.minimum(1.0, 2.5 / np.maximum(norm, 1e-12)))
        residuals[i] = residual * weight[:, None]
        landmark[i] = landmark[i] * weight[:, None, None]
        camera[i] = camera[i] * weight[:, None, None]
    return (
        cost,
        np.concatenate(residuals, axis=1),
        np.concatenate(landmark, axis=1),
        np.concatenate(camera, axis=1),
    )


def conditional_bundle(
    anchors,
    context_pixels,
    context_pose,
    K,
    *,
    initial=None,
    incoming_pixels=None,
    iterations=6,
):
    """Optimize landmarks, optionally sharing one incoming pose through Schur.

    Depth and pixel scales match the preceding conditional noise experiment.
    The six-dimensional reduced matrix diagnoses rank only. Shared historical
    pose/calibration errors are deliberately not declared independent information.
    """
    anchors, context_pixels, K = map(np.asarray, (anchors, context_pixels, K))
    if (
        anchors.ndim != 2
        or anchors.shape[1:] != (3,)
        or len(anchors) < 3
        or context_pixels.shape != (len(anchors), 2)
        or not np.isfinite(anchors).all()
        or not np.isfinite(context_pixels).all()
        or (anchors[:, 2] <= 0).any()
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or not is_rigid(context_pose)
    ):
        raise ValueError(
            "bundle requires finite aligned observations and rigid context"
        )
    if (initial is None) != (incoming_pixels is None):
        raise ValueError("incoming pose and pixels must be supplied together")
    if initial is not None and (
        not is_rigid(initial)
        or np.shape(incoming_pixels) != (len(anchors), 2)
        or not np.isfinite(incoming_pixels).all()
    ):
        raise ValueError("invalid incoming bundle observations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")
    whitening = np.linalg.inv(np.linalg.cholesky(point_covariances(anchors, K)))
    points = anchors.copy()
    pose = None if initial is None else initial.copy()
    diagnostic = {"iterations": 0, "tracks": len(points), "status": "iteration_limit"}
    for iteration in range(iterations):
        cost, residual, Jl, Jp = _weighted(
            points,
            anchors,
            whitening,
            context_pixels,
            context_pose,
            K,
            pose,
            incoming_pixels,
        )
        C = np.einsum("nri,nrj->nij", Jl, Jl)
        bl = np.einsum("nri,nr->ni", Jl, residual)
        inverse = np.linalg.inv(C + np.eye(3) * 1e-6)
        if pose is None:
            dp = np.zeros(6)
            dl = -np.einsum("nij,nj->ni", inverse, bl)
        else:
            A = np.einsum("nri,nrj->ij", Jp, Jp)
            B = np.einsum("nri,nrj->nij", Jp, Jl)
            bp = np.einsum("nri,nr->i", Jp, residual)
            reduced = A - np.einsum("nik,nkl,njl->ij", B, inverse, B)
            gradient = bp - np.einsum("nik,nkl,nl->i", B, inverse, bl)
            scale = np.diag([float(np.median(anchors[:, 2]))] * 3 + [1.0] * 3)
            eigenvalues = np.linalg.eigvalsh(scale @ reduced @ scale)
            ratio = float(eigenvalues[0] / eigenvalues[-1])
            diagnostic["conditional_rank_ratio"] = ratio
            if ratio < 1e-8:
                return (
                    initial.copy(),
                    anchors.copy(),
                    {**diagnostic, "status": "weak_geometry"},
                )
            dp = -np.linalg.solve(reduced + np.diag(np.diag(A)) * 1e-5, gradient)
            dl = -np.einsum("nij,nj->ni", inverse, bl + np.einsum("nji,j->ni", B, dp))
        divisor = max(
            1.0,
            np.linalg.norm(dp[:3]) / 0.1,
            np.linalg.norm(dp[3:]) / np.deg2rad(5),
            np.linalg.norm(dl, axis=1).max() / 0.1,
        )
        dp, dl = dp / divisor, dl / divisor
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            changed = points + fraction * dl
            candidate = None if pose is None else retract(pose, dp * fraction)
            if (changed[:, 2] <= 0.05).any():
                continue
            try:
                new_cost = _weighted(
                    changed,
                    anchors,
                    whitening,
                    context_pixels,
                    context_pose,
                    K,
                    candidate,
                    incoming_pixels,
                )[0]
            except ValueError:
                continue
            if new_cost <= cost:
                pose, points, accepted = candidate, changed, True
                break
        diagnostic.update(
            iterations=iteration + 1,
            initial_cost=diagnostic.get("initial_cost", cost),
            final_cost=new_cost if accepted else cost,
        )
        if not accepted:
            diagnostic["status"] = "no_decreasing_step"
            break
        if max(np.linalg.norm(dp), np.linalg.norm(dl, axis=1).max()) < 1e-6:
            diagnostic["status"] = "step_tolerance"
            break
    return pose, points, diagnostic


def register_multiview(first, second, contexts, calibration, config, *, initial=None):
    """Use one of at most three historical contexts, with a bounded track ledger.

    contexts contains (frame id, native features, T_context_source). Caller owns
    segment/gauge and causality checks. Insufficient context retains the original
    pair result explicitly. Once a multiview fit starts, failure abstains.
    """
    base = initial or register_anchor(first, second, calibration, config)
    if base.T_target_source is None:
        return base
    pairs, _ = match_descriptors(first.descriptors, second.descriptors, config.ratio)
    pairs = pairs[np.isfinite(first.points_R[pairs[:, 0]]).all(axis=1)]
    pair_parity = {int(a): i % 2 for i, (a, _) in enumerate(pairs)}
    choices = []
    for frame, features, transform in contexts[:3]:
        if not is_rigid(transform):
            raise ValueError("context pose must be rigid")
        links, _ = match_descriptors(
            first.descriptors, features.descriptors, config.ratio
        )
        incoming, _ = match_descriptors(
            features.descriptors, second.descriptors, config.ratio
        )
        source_context, context_target = dict(links), dict(incoming)
        triples = np.array(
            [
                (a, source_context[a], b)
                for a, b in pairs
                if a in source_context and context_target.get(source_context[a]) == b
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        if not len(triples):
            continue
        anchors = first.points_R[triples[:, 0]]
        predicted = transform_points(anchors, transform)
        error = np.linalg.norm(
            project_points(predicted, calibration.K_R) - features.pixels[triples[:, 1]],
            axis=1,
        )
        triples = triples[(predicted[:, 2] > 0.05) & (error <= config.reprojection_px)]
        if len(triples) >= config.min_pairs:
            choices.append((len(triples), frame, features, transform, triples))
    if not choices:
        return replace(
            base,
            diagnostics={
                **base.diagnostics,
                "multiview": {
                    "status": "pair_fallback",
                    "reason": "insufficient cycle-consistent context",
                },
            },
        )
    _, frame, context, transform, triples = max(choices, key=lambda v: (v[0], -v[1]))
    # Preserve the original pair split while capping both populations separately.
    parity = np.array([pair_parity[int(a)] for a in triples[:, 0]])
    keep = []
    for group in (0, 1):
        pool = np.flatnonzero(parity == group)
        if len(pool):
            keep.extend(
                pool[np.linspace(0, len(pool) - 1, min(64, len(pool)), dtype=int)]
            )
    triples = triples[sorted(keep)]
    q, c, t = triples.T
    anchors = first.points_R[q]
    parity = np.array([pair_parity[int(a)] for a in q])
    fit, test = np.flatnonzero(parity == 0), np.flatnonzero(parity == 1)
    diagnostic = {
        "method": "conditional_three_view_bundle",
        "context": int(frame),
        "track_rows": triples.tolist(),
        "fit_pairs": len(fit),
        "validation_pairs": len(test),
        "source_depth_factors": len(triples),
        "context_depth_factors": 0,
        "incoming_depth_factors": 0,
        "historical_poses": "conditioned_local_estimates",
    }

    def rejected(reason):
        return Registration(
            None, {**diagnostic, "status": "rejected", "reason": reason}
        )

    if len(fit) < config.min_fit or len(test) < config.min_validation:
        return rejected("insufficient disjoint multiview support")
    diagnostic["fitting_track_indices"] = fit.tolist()
    diagnostic["validation_track_indices"] = test.tolist()
    try:
        # Test landmarks use historical evidence only, never incoming test pixels.
        _, historical, _ = conditional_bundle(
            anchors, context.pixels[c], transform, calibration.K_R
        )
        pose, _, solve = conditional_bundle(
            anchors[fit],
            context.pixels[c[fit]],
            transform,
            calibration.K_R,
            initial=base.T_target_source,
            incoming_pixels=second.pixels[t[fit]],
        )
    except (ValueError, np.linalg.LinAlgError):
        return rejected("conditional bundle numerical failure")
    diagnostic["bundle"] = solve
    if solve["status"] == "weak_geometry":
        return rejected("weak conditional bundle geometry")
    # Original source samples verify the pose as well as historical-only points.
    predicted = transform_points(historical, pose)
    native = transform_points(anchors, pose)
    error = np.linalg.norm(
        project_points(predicted, calibration.K_R) - second.pixels[t], axis=1
    )
    native_error = np.linalg.norm(
        project_points(native, calibration.K_R) - second.pixels[t], axis=1
    )
    measured = second.points_R[t]
    depth = np.isfinite(measured).all(axis=1)
    limits = config.depth_absolute_m + config.depth_relative * np.where(
        depth, measured[:, 2], 0
    )
    valid = (
        (predicted[:, 2] > 0.05)
        & (error <= config.reprojection_px)
        & (native_error <= config.reprojection_px)
    )
    valid &= ~depth | (np.linalg.norm(native - measured, axis=1) <= limits)
    selected = test[valid[test]]
    diagnostic.update(
        fit_inliers=int(valid[fit].sum()),
        validation_inliers=len(selected),
        validation_fraction=float(valid[test].mean()),
        validation_depth_checks=int(depth[selected].sum()),
    )
    if (
        valid[fit].sum() < config.min_fit
        or valid[fit].mean() < config.min_fraction
        or len(selected) < config.min_validation
        or valid[test].mean() < config.min_fraction
    ):
        return rejected("withheld multiview disagreement")
    coverage = min(
        grid_coverage(first.pixels[q[selected]], calibration.rgb_shape),
        grid_coverage(second.pixels[t[selected]], calibration.rgb_shape),
    )
    information = information_ratio(predicted[selected])
    diagnostic.update(
        validation_grid_coverage=coverage,
        validation_information_ratio=information,
        validation_pixel_rmse=float(np.sqrt(np.mean(error[selected] ** 2))),
    )
    if (
        coverage < config.min_grid_coverage
        or information < config.min_information_ratio
    ):
        return rejected("weak withheld multiview geometry")
    return Registration(
        pose,
        {
            **diagnostic,
            "status": "accepted",
            "reason": "withheld three-view verification",
        },
    )
