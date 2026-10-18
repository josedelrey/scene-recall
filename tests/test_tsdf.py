"""Analytic geometry, visibility and resource contracts for projective fusion."""

from dataclasses import replace

import numpy as np
import pytest
from test_mapping import plane, pose

from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, TSDFVolume, fuse_tsdf


def view(index=0, T=None, z=2.0, colored=True):
    T = np.eye(4) if T is None else T
    observation, calibration, _ = plane(T)
    depth = observation.depth + z - 2
    return TSDFView(
        index,
        T,
        depth,
        observation.rgb,
        np.full(depth.shape, colored, dtype=bool),
    ), calibration.K_D


def test_rotated_translated_plane_has_correct_zero_crossing_and_color():
    a, K = view()
    b, _ = view(1, pose(x=0.1, z=0.04, yaw=0.12))
    volume = fuse_tsdf([a, b], K)
    cloud = volume.surface([a, b], K)
    assert len(cloud.points) > 1000
    # Nearest-pixel projective depth is approximate on a slanted camera grid.
    bound = 0.5 * 2.2 / K[0, 0] * np.tan(0.12) + 2e-7
    assert np.max(np.abs(cloud.points[:, 2] - 2)) < bound
    assert cloud.view_support.max() == 2
    np.testing.assert_array_equal(
        cloud.colors[cloud.color_support > 0],
        np.tile([30, 100, 210], ((cloud.color_support > 0).sum(), 1)),
    )
    assert np.isfinite(cloud.signed_residual_std_m).all()


def test_one_view_one_vote_and_sorted_input_order_are_reproducible():
    a, K = view(7)
    b, _ = view(2)
    first, second = fuse_tsdf([a, b], K), fuse_tsdf([b, a], K)
    for field in ("keys", "signed_distance_m", "frame_support"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.frame_support.max() == 2
    with pytest.raises(ValueError, match="distinct"):
        fuse_tsdf([a, a], K)


def test_free_space_removes_minority_foreground_without_occlusion_carving_background():
    foreground, K = view(0, z=1.8)
    background, _ = view(1)
    third, _ = view(2)
    volume = fuse_tsdf([foreground, background, third], K)
    cloud = volume.surface([foreground, background, third], K)
    # The foreground view cannot observe behind its truncation band. Two
    # background views give positive free-space votes around the weak layer.
    np.testing.assert_allclose(cloud.points[:, 2], 2, atol=1e-6)
    center = np.argmin(
        np.linalg.norm((volume.keys + 0.5) * 0.02 - [0, 0, 1.81], axis=1)
    )
    assert volume.signed_distance_m[center] > 0


def test_unknown_depth_does_not_vote_and_rgb_cannot_change_geometry():
    a, K = view()
    empty = replace(a, frame_index=1, depth=np.full(a.depth.shape, np.nan))
    volume = fuse_tsdf([a, empty], K)
    assert volume.frame_support.max() == 1
    original = volume.surface([a, empty], K)
    altered = replace(
        a, colors=np.zeros_like(a.colors), colored=np.zeros_like(a.colored)
    )
    other = fuse_tsdf([altered, empty], K).surface([altered, empty], K)
    np.testing.assert_array_equal(original.points, other.points)
    assert not other.color_support.any()


def test_empty_depth_and_unknown_corners_never_generate_surfaces():
    a, K = view()
    a = replace(a, depth=np.full(a.depth.shape, np.nan))
    volume = fuse_tsdf([a], K)
    assert volume.surface([a], K).points.shape == (0, 3)
    partial = TSDFVolume(
        TSDFConfig(),
        np.zeros(3, dtype=np.int64),
        (2, 2, 2),
        np.array([[0, 0, 0], [1, 0, 0]]),
        np.array([-0.01, 0.01]),
        np.ones(2, dtype=np.uint16),
    )
    assert partial.surface([a], K).points.shape == (0, 3)


def test_grid_and_active_limits_raise_without_silent_pruning():
    a, K = view()
    with pytest.raises(ValueError, match="scratch grid capacity"):
        fuse_tsdf([a], K, TSDFConfig(max_grid_voxels=10))
    with pytest.raises(ValueError, match="active voxel capacity"):
        fuse_tsdf([a], K, TSDFConfig(max_active_voxels=10))


@pytest.mark.parametrize(
    "changes",
    [
        {"voxel_size_m": 0},
        {"truncation_m": np.nan},
        {"truncation_m": 0.01},
        {"max_grid_voxels": True},
        {"max_active_voxels": -1},
    ],
)
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        TSDFConfig(**changes)


def test_invalid_pose_and_calibration_are_rejected():
    a, K = view()
    T = np.eye(4)
    T[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        fuse_tsdf([replace(a, pose=T)], K)
    with pytest.raises(ValueError, match="calibration"):
        fuse_tsdf([a], np.zeros_like(K))


def test_runner_keeps_target_depth_out_of_fusion_and_repeats_exactly(tmp_path):
    import json
    import runpy
    from pathlib import Path

    from PIL import Image
    from test_heldout import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    root = Path(__file__).parents[1]
    run = runpy.run_path(root / "experiments/compare_tsdf_fusion.py")["run"]
    verify = runpy.run_path(root / "experiments/verify_mapping_repeats.py")["verify"]
    outputs = []
    for index, depth in enumerate((2.0, 2.3)):
        folder = tmp_path / str(index)
        folder.mkdir()
        source, odometry = continuous_run(folder, heldout_depth=depth)
        baseline, output = folder / "baseline", folder / "tsdf"
        run_heldout(
            [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
        )
        result = run(baseline, [odometry], output)
        assert result["split"]["train_indices"] == [0, 2]
        assert result["views"][0]["status"] == "evaluated"
        outputs.append(output)
        if index == 0:
            repeat = folder / "repeat"
            run(baseline, [odometry], repeat)
            assert verify(output, repeat)["status"] == "exact_match"
            report = json.loads((repeat / "report.json").read_text())
            report["tsdf_config"]["truncation_m"] = 0.08
            (repeat / "report.json").write_text(json.dumps(report))
            with pytest.raises(ValueError, match="tsdf_config"):
                verify(output, repeat)
        Image.fromarray(np.full((480, 640), 12000, dtype=np.uint16)).save(
            source / "heldout.png"
        )
        with pytest.raises(AssertionError):
            run(baseline, [odometry], folder / "corrupt")
        assert not (folder / "corrupt").exists()
        assert not list(folder.glob(".tsdf-*"))
    for map_name in ("sparse", "gt"):
        for suffix in ("volume", "surface"):
            with (
                np.load(outputs[0] / f"{map_name}_000_{suffix}.npz") as a,
                np.load(outputs[1] / f"{map_name}_000_{suffix}.npz") as b,
            ):
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_runner_preserves_unpaired_chunk_geometry_and_exclusions(tmp_path):
    import runpy
    from pathlib import Path

    from test_heldout import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    _, odometry = continuous_run(tmp_path, missing_reference=0)
    baseline, output = tmp_path / "baseline", tmp_path / "tsdf"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    run = runpy.run_path(
        Path(__file__).parents[1] / "experiments/compare_tsdf_fusion.py"
    )["run"]
    result = run(baseline, [odometry], output)
    assert result["pooled"] == {}
    assert result["views"][0]["status"] == "unavailable_unpaired_training_references"
    with np.load(output / "sparse_000_surface.npz") as archive:
        assert len(archive["points"]) > 0


def test_primary_scores_use_shared_mesh_baseline_pixels_without_disc_control_bias():
    import runpy
    from pathlib import Path

    score = runpy.run_path(
        Path(__file__).parents[1] / "experiments/compare_tsdf_fusion.py"
    )["score_cohort"]
    observed = np.ones(2)
    predictions = {
        "baseline_gt": np.array([1.1, 1.2]),
        "mesh_gt": np.array([1.01, 1.02]),
        "tsdf_gt": np.array([1.01, np.nan]),
    }
    scores, common = score(observed, predictions, np.ones(2, dtype=bool))
    assert common.tolist() == [True, True]
    assert scores["baseline_gt"]["common_all_modes"]["target_samples"] == 2
    assert scores["mesh_gt"]["common_all_modes"]["absolute_depth_error_m"][
        "mean"
    ] == pytest.approx(0.015)
    assert scores["tsdf_gt"]["common_all_modes"]["coverage"] == 0.5
    assert scores["tsdf_gt"]["own"]["missing_penalized_capped_mae_m"][
        "0.2"
    ] == pytest.approx(0.105)


def test_signed_distance_fusion_denoises_across_surface_voxel_boundaries():
    from scene_recall.mapping.voxel import VoxelMap, sample_observation

    observation, calibration, config = plane()
    rng = np.random.default_rng(23)
    baseline = VoxelMap(config.voxel_size_m)
    views = []
    for index in range(8):
        noisy = replace(
            observation,
            depth=(
                observation.depth + rng.normal(0, 0.005, observation.depth.shape)
            ).astype(np.float32),
        )
        samples = sample_observation(noisy, calibration, config)
        baseline.integrate(samples, np.eye(4), index)
        views.append(TSDFView(index, np.eye(4), samples.depth))
    surface = fuse_tsdf(views, calibration.K_D).surface(views, calibration.K_D)
    before = np.abs(baseline.cloud().points[:, 2] - 2).mean()
    after = np.abs(surface.points[:, 2] - 2).mean()
    # The true plane lies on a voxel boundary. Baseline averaging keeps
    # positive/negative noise in separate cells. The scalar field fuses
    # evidence across that boundary without discarding either sign.
    assert len(surface.points) > 1000
    assert after < before / 2


def test_synthetic_pose_bias_reverses_external_geometry_and_self_consistency():
    import runpy
    from pathlib import Path

    probe = runpy.run_path(
        Path(__file__).parents[1] / "experiments/diagnose_tsdf_pose_bias.py"
    )["probe"]
    exact, _ = probe(0)
    biased, _ = probe(0.08)
    for score in exact["scores"].values():
        assert score["absolute_depth_error_m"]["mean"] == pytest.approx(0, abs=1e-12)
    errors = {
        name: score["absolute_depth_error_m"]["mean"]
        for name, score in biased["center_scores"].items()
    }
    assert errors["baseline_reference"] == pytest.approx(0, abs=1e-12)
    assert errors["tsdf_reference"] == pytest.approx(0.08, abs=1e-12)
    assert errors["baseline_estimated"] == pytest.approx(0.08, abs=1e-12)
    assert errors["tsdf_estimated"] == pytest.approx(0, abs=1e-12)
    assert (
        biased["scores"]["tsdf_estimated"]["absolute_depth_error_m"]["mean"]
        < biased["scores"]["baseline_estimated"]["absolute_depth_error_m"]["mean"]
    )
