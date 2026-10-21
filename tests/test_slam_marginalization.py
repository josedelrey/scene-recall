"""Information, ownership and gauge contracts for variable retirement."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.fixed_landmarks import make_direct_factors
from scene_recall.multiview.joint_ba import landmark_layout
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.slam.graph import linear_system, marginalize_oldest
from scene_recall.slam.marginal import (
    eliminate_columns,
    marginalize,
    prior_supports_camera,
)
from scene_recall.slam.system import SLAMConfig

scene = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
)["scene"]


def subset(factors, mask):
    return replace(
        factors,
        **{name: getattr(factors, name)[mask] for name in factors.__dataclass_fields__},
    )


@pytest.mark.parametrize("model", ["native", "axial"])
def test_square_root_camera_retirement_matches_existing_schur(model):
    calibration, tracks, samples, _ = scene(5, exact=False)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(5)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    poses = np.array([f.T_SR for f in tracks.frames])
    config = BatchConfig(depth_model=model)
    old = new = None
    for boundary in range(3):
        keys = list(range(boundary, 5))
        root = 0 if boundary == 0 else None
        old, keep, _ = marginalize_oldest(
            poses, factors, calibration.K_R, keys, root, old, config
        )
        new, mask, _, info = marginalize(
            poses, factors, calibration.K_R, keys, root, new, config, cameras=(keys[0],)
        )
        np.testing.assert_array_equal(mask, keep)
        np.testing.assert_allclose(
            new.A.T @ new.A, old.A.T @ old.A, atol=2e-6, rtol=2e-7
        )
        np.testing.assert_allclose(
            new.A.T @ new.b, old.A.T @ old.b, atol=2e-7, rtol=1e-6
        )
        assert info["consumed_observations"] == int((~keep).sum())
        factors = replace(subset(factors, keep), view=factors.view[keep] - 1)
        poses = poses[1:]


def test_point_retirement_keeps_camera_coupling_and_consumes_only_incident_rows():
    calibration, tracks, samples, _ = scene(5, exact=False)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(5)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    poses = np.array([f.T_SR for f in tracks.frames])
    ids, _, points = landmark_layout(factors)
    config = BatchConfig(loss="linear")
    fingerprint = factors.fingerprint()
    _, H, g = linear_system(
        poses, points, ids, factors, calibration.K_R, config, list(range(5)), None
    )
    selected = ids[ids % 3 == 0]
    prior, keep, retired, _ = marginalize(
        poses,
        factors,
        calibration.K_R,
        list(range(5)),
        0,
        None,
        config,
        tracks=selected,
    )
    np.testing.assert_array_equal(retired, selected)
    np.testing.assert_array_equal(keep, ~np.isin(factors.track_id, selected))
    raw = subset(factors, keep)
    live, _, remaining = landmark_layout(raw)
    _, combined, gradient = linear_system(
        poses, remaining, live, raw, calibration.K_R, config, list(range(5)), prior
    )
    eliminate = np.array(
        [
            30 + 3 * i + j
            for i, key in enumerate(ids)
            if key in selected
            for j in range(3)
        ]
    )
    retained = np.setdiff1d(np.arange(len(g)), eliminate)
    cross = H[np.ix_(retained, eliminate)]
    expected = H[np.ix_(retained, retained)] - cross @ np.linalg.solve(
        H[np.ix_(eliminate, eliminate)], cross.T
    )
    expected_g = g[retained] - cross @ np.linalg.solve(
        H[np.ix_(eliminate, eliminate)], g[eliminate]
    )
    np.testing.assert_allclose(combined, expected, atol=2e-7, rtol=1e-8)
    np.testing.assert_allclose(gradient, expected_g, atol=2e-8, rtol=1e-7)
    assert np.linalg.norm((prior.A.T @ prior.A)[:6, 6:12]) > 1
    assert prior_supports_camera(prior, 1)
    assert not prior_supports_camera(prior, 9)
    assert factors.fingerprint() == fingerprint
    # Reuse a prior containing camera-camera and camera-point terms across
    # both a conditioned root boundary and a free-camera boundary.
    for boundary in range(2):
        keys = list(range(boundary, 5))
        root = 0 if boundary == 0 else None
        prior, keep, retired, _ = marginalize(
            poses, raw, calibration.K_R, keys, root, prior, config, cameras=(keys[0],)
        )
        assert len(retired) == 0
        if root == 0:
            expected, expected_g = combined[6:, 6:], gradient[6:]
        else:
            expected = combined[6:, 6:] - combined[6:, :6] @ np.linalg.solve(
                combined[:6, :6], combined[:6, 6:]
            )
            expected_g = gradient[6:] - combined[6:, :6] @ np.linalg.solve(
                combined[:6, :6], gradient[:6]
            )
        raw = replace(subset(raw, keep), view=raw.view[keep] - 1)
        poses = poses[1:]
        _, combined, gradient = linear_system(
            poses, remaining, live, raw, calibration.K_R, config, keys[1:], prior
        )
        np.testing.assert_allclose(combined, expected, atol=4e-7, rtol=2e-7)
        np.testing.assert_allclose(gradient, expected_g, atol=3e-8, rtol=2e-7)


def test_rank_deficient_retirement_matches_least_squares_without_false_information():
    rng = np.random.default_rng(27)
    old = rng.normal(size=(50, 4))
    old = np.column_stack((old, old[:, 0] + 2 * old[:, 2], np.zeros(50)))
    retained = rng.normal(size=(50, 8))
    residual = rng.normal(size=50)
    J = np.column_stack((old, retained))
    A, b, rank = eliminate_columns(J, residual, np.arange(6, 14), np.arange(6))
    projector = np.eye(50) - old @ np.linalg.pinv(old)
    np.testing.assert_allclose(A.T @ A, retained.T @ projector @ retained, atol=1e-12)
    np.testing.assert_allclose(A.T @ b, retained.T @ projector @ residual, atol=1e-12)
    assert rank == 4
    # Projection roundoff must not become information through equilibration.
    J = np.column_stack((old, old[:, :4]))
    A, b, _ = eliminate_columns(J, residual, np.arange(6, 10), np.arange(6))
    assert A.shape == (0, 4)
    assert b.shape == (0,)


def test_camera_without_raw_rows_can_retire_from_prior_alone():
    from scene_recall.slam.graph import square_root_prior

    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    prior = square_root_prior([0, 1, 2], poses, np.eye(18), np.zeros(18))
    result, keep, retired, info = marginalize(
        poses, None, np.eye(3), [0, 1, 2], None, prior, BatchConfig(), cameras=(0,)
    )
    assert result.camera_ids == (1, 2)
    assert len(keep) == len(retired) == 0
    assert info["eliminated_rank"] == 6
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(12), atol=1e-12)


def test_lifecycle_configuration_is_explicit_and_preserves_legacy_default():
    assert SLAMConfig().marginalization == "schur"
    assert SLAMConfig().stale_keyframes == 0
    for arguments in (
        {"stale_keyframes": True},
        {"stale_keyframes": -1},
        {"stale_keyframes": 2},
        {"marginalization": "unknown"},
    ):
        with pytest.raises(ValueError):
            SLAMConfig(**arguments)


def test_persistent_admission_preserves_frontend_and_raw_ownership(monkeypatch):
    from types import SimpleNamespace

    from scene_recall.data import Observation
    from scene_recall.multiview.temporal import PairMotion

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from slam_persistent import PersistentSeedsSLAM, persistent_seeds

    calibration, tracks, samples, _ = scene(2, exact=True, skew=0)
    features = tracks.frames[0].features
    owners = np.arange(len(features.pixels))
    lengths = {i: 1 + i % 5 for i in owners}
    engine = PersistentSeedsSLAM(SLAMConfig(max_landmarks=20, exposure="static"))
    engine.frontend = SimpleNamespace(
        owners={0: owners},
        landmarks={
            i: SimpleNamespace(observations=[None] * lengths[i]) for i in owners
        },
    )
    monkeypatch.setattr("slam_persistent.bind_depth", lambda *_: samples[0])
    observation = Observation(
        0,
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.ones((240, 320), dtype=np.float32),
    )
    engine._ingest(
        observation,
        calibration,
        np.eye(4),
        features,
        PairMotion(np.eye(4), "synchronized", True),
    )
    assert engine.frontend.owners[0] is owners
    selected = persistent_seeds(
        features.pixels, owners, lengths, 20, calibration.rgb_shape
    )
    assert set(engine.last_measurement["feature_index"]) == set(selected)
    assert len(engine.last_measurement["native_index"]) == 20
    assert len(np.unique(engine.last_measurement["native_index"])) == 20
    np.testing.assert_array_equal(engine.last_measurement["view"], np.zeros(20))
    # Within one cell, mature observations win. Equal histories use original IDs.
    pixels = np.full((4, 2), 10.0)
    assert persistent_seeds(
        pixels, range(4), {0: 1, 1: 3, 2: 3, 3: 2}, 2, (240, 320)
    ) == [1, 2]
    assert persistent_seeds(pixels, [], {}, 20, (240, 320)) == []
    missing = replace(
        samples[0],
        source_index=np.full(len(owners), -1),
        points_D=np.full_like(samples[0].points_D, np.nan),
        points_R=np.full_like(samples[0].points_R, np.nan),
    )
    empty = PersistentSeedsSLAM(SLAMConfig(max_landmarks=20, exposure="static"))
    empty.frontend = engine.frontend
    monkeypatch.setattr("slam_persistent.bind_depth", lambda *_: missing)
    result = empty._ingest(
        observation,
        calibration,
        np.eye(4),
        features,
        PairMotion(np.eye(4), "unknown_clock", False),
    )
    assert result["reason"] == "no admissible measurements"
    assert not empty.points
