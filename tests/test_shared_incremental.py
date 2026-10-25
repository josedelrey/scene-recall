"""Incremental derivative reuse preserves the owned nonlinear likelihood."""

from dataclasses import replace

import numpy as np
import pytest
from test_shared_estimation import objective, problem

from scene_recall.geometry.camera import project_points, transform_points
from scene_recall.multiview.audit import normalized_spectrum
from scene_recall.multiview.batch_ba import compact_blocks, compact_linearize
from scene_recall.multiview.fixed_landmarks import DirectFactors, direct_cost
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.slam.observation_welding import WeldingConfig
from scene_recall.slam.shared_estimation import SharedConfig
from scene_recall.slam.shared_linearization import (
    ObservationCache,
    RelinearizationConfig,
)
from scene_recall.slam.shared_preconditioner import CoarsePreconditioner
from scene_recall.slam.shared_solver import SchurOperator, solve_shared


@pytest.mark.parametrize("model", ["native", "axial"])
def test_cached_residual_and_fresh_derivatives_match_reference(model):
    poses, factors, K, _ = objective()
    _, inverse, points = landmark_layout(factors)
    cache = ObservationCache()
    cache.bind(factors, K, model)
    actual = cache.linearize(poses, points, inverse)
    reference = compact_linearize(poses, points, factors, K, inverse, model)
    for a, b in zip(actual, reference, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-11)
    cached = cache.linearize(poses, points, inverse)
    assert cache.diagnostic()["reused_fraction"] == 0.5
    for a, b in zip(actual, cached, strict=True):
        np.testing.assert_array_equal(a, b)


def test_cache_refreshes_both_endpoints_and_measurement_edits():
    poses, factors, K, _ = objective()
    _, inverse, points = landmark_layout(factors)
    cache = ObservationCache()
    cache.bind(factors, K, "native")
    cache.linearize(poses, points, inverse)
    # A point edit must refresh every camera observing that point, including
    # cameras whose own pose did not move.
    points[0, 0] += 0.01
    poses[1, 1, 3] += 0.01
    actual = cache.linearize(poses, points, inverse)
    reference = compact_linearize(poses, points, factors, K, inverse)
    mask = (inverse == 0) | (factors.view == 1)
    for a, b in zip(actual[1:], reference[1:], strict=True):
        np.testing.assert_allclose(a[mask], b[mask], atol=2e-11)
    factors = replace(factors, whitening_D=factors.whitening_D * 2)
    cache.bind(factors, K, "native")
    assert not cache.valid.any()
    cache.linearize(poses, points, inverse)
    # Row order and deletion may change without creating a second factor owner.
    order = np.arange(len(factors.view))[::-2]
    reduced = replace(
        factors,
        **{
            name: getattr(factors, name)[order] for name in factors.__dataclass_fields__
        },
    )
    cache.bind(reduced, K, "native")
    assert cache.valid.all()
    cache.bind(reduced, K * 1.01, "native")
    assert not cache.valid.any()


def test_age_forces_relinearization_without_a_state_change():
    poses, factors, K, _ = objective()
    _, inverse, points = landmark_layout(factors)
    cache = ObservationCache(RelinearizationConfig(max_age=1))
    cache.bind(factors, K, "native")
    cache.linearize(poses, points, inverse)
    cache.linearize(poses, points, inverse)
    assert not cache.last_fresh
    cache.linearize(poses, points, inverse)
    assert cache.last_fresh


def test_subthreshold_movements_accumulate_from_last_refresh():
    poses, factors, K, _ = objective()
    _, inverse, points = landmark_layout(factors)
    cache = ObservationCache(RelinearizationConfig(max_age=100))
    cache.bind(factors, K, "native")
    cache.linearize(poses, points, inverse)
    initial = cache.arrays["pose"].copy()
    for _ in range(2):
        poses[1, 0, 3] += 0.0009
        cache.linearize(poses, points, inverse)
    np.testing.assert_array_equal(cache.arrays["pose"], initial)
    poses[1, 0, 3] += 0.0009
    cache.linearize(poses, points, inverse)
    mask = factors.view == 1
    np.testing.assert_array_equal(cache.arrays["pose"][mask], poses[factors.view[mask]])


@pytest.mark.parametrize("fixed", [(0,), (0, 3), (0, 1, 2, 3, 4)])
def test_coarse_galerkin_system_and_linear_step_match_dense(fixed):
    poses, factors, K, config = objective()
    ids, inverse, points = landmark_layout(factors)
    r, JP, JL = compact_linearize(poses, points, factors, K, inverse)
    _, weights = direct_cost(r, factors, config)
    blocks = compact_blocks(
        r, JP, JL, weights, inverse, factors.view, len(ids), len(poses)
    )
    operator = SchurOperator(blocks, factors.view, inverse, fixed, 0.003)
    coarse = CoarsePreconditioner(operator, group_size=2)
    n = len(operator.rhs)
    S = (
        np.column_stack([operator.multiply(row) for row in np.eye(n)])
        if n
        else np.zeros((0, 0))
    )
    Z = np.zeros((n, len(coarse.matrix)))
    for i, group in enumerate(coarse.groups):
        Z[i * 6 : i * 6 + 6, group * 6 : group * 6 + 6] = np.eye(6)
    np.testing.assert_allclose(coarse.matrix, Z.T @ S @ Z, rtol=2e-10, atol=1e-8)
    if n:
        assert (
            np.linalg.eigvalsh(
                np.column_stack([coarse(row) for row in np.eye(n)])
            ).min()
            > 0
        )
    dp, diagnostic = operator.solve(preconditioner="coarse", tolerance=1e-10)
    expected = np.linalg.solve(S, operator.rhs) if n else np.array([])
    np.testing.assert_allclose(dp[operator.free].ravel(), expected, atol=2e-8)
    assert diagnostic["relative_residual"] <= 2e-10


def test_zero_threshold_matches_complete_nonlinear_solver():
    poses, factors, K, config = objective()
    reference = solve_shared(poses, factors, K, (0,), config)
    cache = ObservationCache(RelinearizationConfig(0, 0, 0))
    actual = solve_shared(poses, factors, K, (0,), config, cache=cache)
    np.testing.assert_allclose(actual[0], reference[0], atol=2e-8)
    np.testing.assert_allclose(actual[1], reference[1], atol=2e-8)


def test_selective_steps_decrease_exact_complete_objective():
    poses, factors, K, config = objective()
    cache = ObservationCache(RelinearizationConfig(0.01, 1, 0.01))
    revised, points, report = solve_shared(
        poses, factors, K, (0,), config, cache=cache, preconditioner="coarse"
    )
    _, inverse, _ = landmark_layout(factors)
    r, _, _ = compact_linearize(revised, points, factors, K, inverse)
    assert direct_cost(r, factors, config)[0] == pytest.approx(
        report["final_cost"], abs=1e-8
    )
    costs = [report["initial_cost"], *report["accepted_costs"]]
    assert np.all(np.diff(costs) < 0)
    assert report["linearization"]["reused_fraction"] > 0
    np.testing.assert_array_equal(revised[0], poses[0])


def test_identity_edits_discard_cached_derivatives():
    graph, estimator = problem()
    estimator.shared = SharedConfig(linearization="selective", preconditioner="coarse")
    estimator.refine(graph)
    cache = estimator.cache
    fingerprints = {
        key: view.factors.fingerprint() for key, view in estimator.views.items()
    }
    estimator.refine(graph)
    assert estimator.cache is cache
    graph.revoke(0, 4, "test split")
    estimator.revoke(0, 4)
    estimator.refine(graph)
    assert estimator.cache is not cache
    assert len(set(graph.components.values())) == 2
    assert fingerprints == {
        key: view.factors.fingerprint() for key, view in estimator.views.items()
    }


def test_covisibility_admission_preserves_recent_camera_support():
    _, factors, _, _ = objective()
    _, prototype = problem()
    from scene_recall.slam.shared_estimation import SharedEstimator

    ledgers = [
        SharedEstimator(
            prototype.calibration,
            WeldingConfig(max_observations_per_view=8),
            SharedConfig(admission=policy),
        )
        for policy in ("balanced", "covisibility")
    ]
    for frame, owners in enumerate((np.arange(8), np.arange(8, 16), np.arange(16))):
        n = len(owners)
        measurement = {
            name: getattr(factors, name)[:n].copy()
            for name in factors.__dataclass_fields__
        }
        measurement.update(
            view=np.full(n, frame, dtype=int),
            pixels=np.zeros((n, 2)),
            frontend_identity=owners,
            feature_index=np.arange(n),
            native_index=np.arange(n),
        )
        for ledger in ledgers:
            ledger.admit(frame, 0, measurement)
    assert ledgers[0].admission_report["previous_shared_tracks"] == 0
    assert ledgers[1].admission_report["previous_shared_tracks"] >= 6
    for ledger in ledgers:
        packet = ledger.views[2]
        assert len(packet.factors.view) == 8
        assert len(np.unique(packet.native_index)) == 8
        assert not packet.factors.native_points.flags.writeable


def test_linear_failure_retries_algorithmic_damping_without_a_prior(monkeypatch):
    poses, factors, K, config = objective()
    original = SchurOperator.solve
    calls = 0

    def transient(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("controlled transient")
        return original(self, **kwargs)

    monkeypatch.setattr(SchurOperator, "solve", transient)
    _, _, report = solve_shared(
        poses, factors, K, (0,), config, retry_linear_failure=True
    )
    assert report["status"] == "refined"
    assert report["linear_failures"] == 1
    assert report["final_cost"] < report["initial_cost"]
    calls = 0
    revised, _, report = solve_shared(poses, factors, K, (0,), config)
    assert report["status"] == "rejected"
    np.testing.assert_array_equal(revised, poses)


@pytest.mark.parametrize("retry", [False, True])
def test_late_linear_failure_preserves_the_declared_transaction_policy(
    monkeypatch, retry
):
    poses, factors, K, config = objective()
    _, inverse, seeds = landmark_layout(factors)
    original, calls = SchurOperator.solve, 0

    def fail_after_first(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise np.linalg.LinAlgError("controlled late failure")
        return original(self, **kwargs)

    fingerprint = factors.fingerprint()
    monkeypatch.setattr(SchurOperator, "solve", fail_after_first)
    revised, points, report = solve_shared(
        poses,
        factors,
        K,
        (0,),
        config,
        cache=ObservationCache(),
        retry_linear_failure=retry,
    )
    if retry:
        assert report["status"] == "refined"
        assert report["linear_failures"] >= 10
        residual, _, _ = compact_linearize(revised, points, factors, K, inverse)
        assert direct_cost(residual, factors, config)[0] == pytest.approx(
            report["final_cost"], abs=1e-8
        )
    else:
        assert report["status"] == "rejected"
        np.testing.assert_array_equal(revised, poses)
        np.testing.assert_array_equal(points, seeds)
    assert fingerprint == factors.fingerprint()


def test_strict_rank_policy_uses_current_derivatives():
    poses, factors, K, config = objective()
    cache = ObservationCache(RelinearizationConfig(1, 30, 1))
    _, _, report = solve_shared(
        poses, factors, K, (0,), config, cache=cache, require_full_rank=True
    )
    assert report["status"] == "refined"
    assert report["rank_check"]["jacobians_current"]
    assert report["linearization"]["reused_fraction"] == 0


def test_well_conditioned_camera_blocks_can_hide_an_unobservable_island():
    # Two internally supported camera groups share one 3D point. Their
    # relative rotation about that point remains unobservable globally.
    rng = np.random.default_rng(97)
    points = rng.uniform([-0.6, -0.5, 2], [0.6, 0.5, 3], (25, 3))
    tracks = np.concatenate(
        (np.arange(12), np.arange(13), np.arange(12, 25), np.arange(13, 25))
    )
    views = np.repeat(np.arange(4), (12, 13, 13, 12))
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.arange(4) * 0.04
    native = np.array(
        [
            transform_points(points[track], np.linalg.inv(poses[view]))
            for view, track in zip(views, tracks, strict=True)
        ]
    )
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    factors = DirectFactors(
        views,
        tracks,
        points[tracks],
        project_points(native, K),
        native,
        np.repeat(np.eye(4)[None], len(views), axis=0),
        np.repeat((np.eye(3) * 40)[None], len(views), axis=0),
        np.ones(len(views)),
    )
    ids, inverse, seeds = landmark_layout(factors)
    r, JP, JL = compact_linearize(poses, seeds, factors, K, inverse)
    _, weights = direct_cost(r, factors, replace(objective()[3], loss="linear"))
    blocks = compact_blocks(r, JP, JL, weights, inverse, views, len(ids), 4)
    operator = SchurOperator(blocks, views, inverse, (0,), 0)
    for diagonal in operator.diagonal:
        assert normalized_spectrum(diagonal)["condition"] < 1e8
    S = np.column_stack([operator.multiply(row) for row in np.eye(len(operator.rhs))])
    assert normalized_spectrum(S)["nullity"] == 3


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_relinearization_threshold(value):
    with pytest.raises(ValueError):
        RelinearizationConfig(point_m=value)


@pytest.mark.parametrize("field", ["linearization", "preconditioner", "admission"])
def test_unknown_incremental_policy(field):
    with pytest.raises(ValueError):
        SharedConfig(**{field: "unknown"})
