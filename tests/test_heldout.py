"""Held-out populations, visibility, pose gauges and end-to-end reproducibility."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers.mapping import plane, pose, row, samples
from helpers.runs import continuous_run, make_run
from PIL import Image

from scene_recall.mapping.evaluation import heldout_metrics, render_depth
from scene_recall.mapping.heldout import (
    check_cohort,
    heldout_split,
    paired_blocks,
    prediction_poses,
    run_heldout,
    score_predictions,
)
from scene_recall.mapping.pipeline import Keyframe, build_submap
from scene_recall.mapping.voxel import MappingConfig, VoxelMap
from scene_recall.odometry.io import load_trajectory


def cloud(points):
    mapping = VoxelMap(0.001)
    mapping.integrate(samples(points), np.eye(4), 0)
    return mapping.cloud()


def test_renderer_nearest_surface_no_observed_depth_and_metric_footprint():
    K = np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]], dtype=float)
    geometry = cloud([[0, 0, 2], [0, 0, 1], [0, 0, -1], [0, 0, 0.01]])
    rendered = render_depth(geometry, np.eye(4), K, (5, 5), 0.11)
    assert rendered[2, 2] == 1  # Foreground hides background and behind-camera points.
    assert rendered[2, 3] == 1  # 10 cm away in the foreground plane.
    assert np.isnan(rendered[3, 3])  # Disc corners do not become square splats.
    assert np.isnan(rendered[0, 0])
    moved = render_depth(geometry, pose(z=-1), K, (5, 5), 0.01)
    assert moved[2, 2] == pytest.approx(1.01)  # Inverse camera-to-map transform.
    assert np.isnan(render_depth(cloud([]), np.eye(4), K, (5, 5), 0.1)).all()


def test_renderer_matches_brute_force_rays_and_offscreen_footprints():
    K = np.array([[20, 0, 3], [0, 15, 2], [0, 0, 1]], dtype=float)
    points = [[-0.35, 0, 2], [0.1, 0.1, 1], [0.2, 0, 3]]
    actual = render_depth(cloud(points), np.eye(4), K, (5, 7), 0.2)
    expected = np.full((5, 7), np.nan)
    for v in range(5):
        for u in range(7):
            hits = [
                z
                for x, y, z in points
                if ((u - 3) * z / 20 - x) ** 2 + ((v - 2) * z / 15 - y) ** 2 <= 0.2**2
            ]
            if hits:
                expected[v, u] = min(hits)
    np.testing.assert_allclose(actual, expected)


def test_rotated_plane_rendering_uses_the_evaluation_lattice_origin():
    observation, calibration, config = plane()
    from scene_recall.mapping.voxel import sample_observation

    mapping = VoxelMap(config.voxel_size_m)
    mapping.integrate(
        sample_observation(observation, calibration, config), np.eye(4), 0
    )
    K = calibration.K_D.copy()
    K[:2] /= 2
    rendered = render_depth(mapping.cloud(), pose(yaw=0.05), K, (17, 17), 0.04)
    v, u = np.indices(rendered.shape)
    rays = np.stack(
        ((u * 2 - 16) / 40, (v * 2 - 16) / 40, np.ones(rendered.shape)), axis=-1
    )
    expected = 2 / (rays @ pose(yaw=0.05)[:3, :3].T)[..., 2]
    np.testing.assert_allclose(
        rendered[np.isfinite(rendered)], expected[np.isfinite(rendered)], atol=0.005
    )


def test_accuracy_coverage_signed_errors_and_empty_populations():
    observed = np.ones(5)
    predicted = np.array([1, 0.8, 1.3, np.nan, 0])
    metric = heldout_metrics(observed, predicted, np.ones(5, dtype=bool), (0.05,))
    assert metric["coverage"] == 3 / 5
    assert metric["missing_samples"] == 2
    assert metric["absolute_depth_error_m"]["mean"] == pytest.approx(1 / 6)
    assert metric["thresholds"]["0.05"] == {
        "agree_samples": 1,
        "front_samples": 1,
        "behind_samples": 1,
        "agreement_of_predicted": 1 / 3,
        "explained_fraction_of_target": 1 / 5,
    }
    empty = heldout_metrics(observed, predicted, np.zeros(5, dtype=bool))
    assert empty["coverage"] is None
    assert empty["absolute_depth_error_m"]["mean"] is None
    with pytest.raises(ValueError, match="valid observed"):
        heldout_metrics(np.array([np.nan]), np.ones(1), np.ones(1, dtype=bool))


def test_common_masks_prevent_population_reversal_and_preserve_missing():
    observed = np.ones(4)
    predictions = {
        "sparse_reference_pose": np.array([1.01, 2, 2, 2]),
        "hybrid_reference_pose": np.array([1.02, np.nan, np.nan, np.nan]),
        "sparse_estimated_pose": np.array([1.03, 2, 2, 2]),
        "hybrid_estimated_pose": np.array([1.04, np.nan, np.nan, np.nan]),
        "gt": np.ones(4),
    }
    scores, masks = score_predictions(observed, predictions)
    a, b = [scores[f"{s}_reference_pose"] for s in ("sparse", "hybrid")]
    assert (
        a["own"]["absolute_depth_error_m"]["mean"]
        > b["own"]["absolute_depth_error_m"]["mean"]
    )
    assert (
        a["common_reference_pose"]["absolute_depth_error_m"]["mean"]
        < b["common_reference_pose"]["absolute_depth_error_m"]["mean"]
    )
    assert b["own"]["coverage"] == 0.25
    assert b["own"]["thresholds"]["0.05"]["explained_fraction_of_target"] == 0.25
    np.testing.assert_array_equal(masks["all_modes"], [True, False, False, False])
    predictions["gt"][:] = np.nan
    scores, masks = score_predictions(observed, predictions)
    assert not masks["all_modes"].any()
    assert (
        scores["hybrid_reference_pose"]["common_all_modes"]["absolute_depth_error_m"][
            "rmse"
        ]
        is None
    )


def test_segment_split_has_no_leakage_or_chunk_gap_assignment():
    rows = [
        row(
            i,
            segment=int(i >= 7),
            reference=np.eye(4),
            status="lost" if i == 7 else None,
        )
        for i in range(14)
    ]
    config = MappingConfig(keyframe_stride=4, max_keyframes_per_submap=2)
    split = heldout_split(rows, config, 2)
    assert split["chunks"] == [[0, 4], [6], [7, 11], [13]]
    assert split["target_indices_by_chunk"] == [[2], [], [9], []]
    assert split["excluded_outside_chunk_extent"] == []
    assert not set(split["train_indices"]) & {
        i for group in split["target_indices_by_chunk"] for i in group
    }
    for offset in (0, 4, True):
        with pytest.raises(ValueError, match="offset"):
            heldout_split(rows, config, offset)
    split = heldout_split([row(i) for i in range(12)], config, 1)
    assert split["excluded_outside_chunk_extent"] == [5]


def test_reference_pose_and_estimated_pose_experiments_measure_different_errors():
    config = MappingConfig(keyframe_stride=2)
    first = row(0, reference=np.eye(4))
    submap = build_submap([Keyframe(first, samples([[0, 0, 2]]))], 0, config)
    target = row(1, T=pose(z=0.2), reference=pose(z=0.1))
    poses = prediction_poses({"sparse": submap}, {"sparse": target})
    assert poses["sparse_reference_pose"][1][2, 3] == 0.1
    assert poses["sparse_estimated_pose"][1][2, 3] == 0.2
    with pytest.raises(ValueError, match="disconnected"):
        prediction_poses({"sparse": submap}, {"sparse": replace(target, segment_id=1)})
    with pytest.raises(ValueError, match="valid target"):
        prediction_poses(
            {"sparse": submap}, {"sparse": replace(target, reference_pose=None)}
        )


def test_real_adapter_repeatability_no_fusion_leakage_and_atomicity(tmp_path):
    source, run = continuous_run(tmp_path)
    config = MappingConfig(pixel_stride=32, keyframe_stride=2)
    outputs = [tmp_path / "a", tmp_path / "b"]
    reports = [run_heldout([run], p, config, 1, (0.02,)) for p in outputs]
    assert reports[0]["split"]["train_indices"] == [0, 2]
    assert reports[0]["summaries"]["0.02"]["evaluated_views"] == 1
    assert reports[0]["views"] == reports[1]["views"]
    verifier = runpy.run_path(
        Path(__file__).parents[1] / "experiments/verify_mapping_repeats.py"
    )["verify"]
    assert verifier(*outputs)["status"] == "exact_match"
    with pytest.raises(ValueError, match="already exists"):
        run_heldout([run], outputs[0], config, 1)
    # Corrupt only the withheld depth to exercise full-input verification.
    Image.fromarray(np.full((480, 640), 11500, dtype=np.uint16)).save(
        source / "heldout.png"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        run_heldout([run], tmp_path / "corrupt", config, 1)
    assert not (tmp_path / "corrupt").exists()
    assert not list(tmp_path.glob(".heldout-*"))


def test_cohort_mismatch_and_missing_references_are_explicit(tmp_path):
    _, run = make_run(tmp_path)
    report, frames = load_trajectory(run)
    config = MappingConfig(keyframe_stride=2)
    loaded = {
        "sparse": (report, frames),
        "hybrid": (report, [replace(f, segment_id=0) for f in frames]),
    }
    with pytest.raises(ValueError, match="trajectory status|segments|partition"):
        check_cohort(loaded, config, 1)
    # Native disconnected fixture has no eligible interior target across its reset.
    result = run_heldout([run], tmp_path / "disconnected", config, 1)
    assert result["summaries"]["0.01"]["evaluated_views"] == 0
    assert result["summaries"]["0.01"]["pooled"] == {}


def test_paired_block_statistics_are_deterministic_and_never_zero_fill():
    rows = []
    for i in range(24):
        predictions = {
            "sparse_reference_pose": np.array([1.1]),
            "hybrid_reference_pose": np.array([1.01]),
            "gt": np.ones(1),
        }
        scores, _ = score_predictions(np.ones(1), predictions)
        rows.append({"segment_id": i // 12, "scores": scores})
    first = paired_blocks(rows, "sparse_reference_pose", "hybrid_reference_pose")
    assert first == paired_blocks(
        rows, "sparse_reference_pose", "hybrid_reference_pose"
    )
    assert first["common_mae_m"]["mean_block_delta"] == pytest.approx(-0.09)
    assert first["common_mae_m"]["negative_delta_views"] == 24
    assert first["common_mae_m"]["block_count"] == 4


def test_withheld_depth_never_enters_map_geometry(tmp_path):
    outputs = []
    for index, depth in enumerate((2.0, 2.3)):
        folder = tmp_path / str(index)
        folder.mkdir()
        _, run = continuous_run(folder, heldout_depth=depth)
        output = folder / "evaluation"
        run_heldout([run], output, MappingConfig(pixel_stride=32, keyframe_stride=2), 1)
        outputs.append(output)
    for name in ("sparse_000_estimated.npz", "sparse_000_reference.npz"):
        with np.load(outputs[0] / name) as a, np.load(outputs[1] / name) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    reports = [json.loads((p / "report.json").read_text()) for p in outputs]
    assert reports[0]["views"] != reports[1]["views"]


@pytest.mark.parametrize(
    "missing_index,status",
    [
        (0, "unavailable_unpaired_training_references"),
        (1, "unavailable_target_reference"),
    ],
)
def test_missing_references_withhold_controlled_scores_without_removing_maps(
    tmp_path, missing_index, status
):
    _, run = continuous_run(tmp_path, missing_reference=missing_index)
    output = tmp_path / "evaluation"
    report = run_heldout(
        [run], output, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    assert report["views"]["0.01"][0]["status"] == status
    assert report["summaries"]["0.01"]["unavailable_status_counts"] == {status: 1}
    assert report["submaps"][0]["keyframe_indices"] == [0, 2]
    with np.load(output / "sparse_000_estimated.npz") as data:
        assert len(data["points"]) > 0


def test_reference_fusion_is_exactly_independent_of_chunk_estimated_anchor():
    config = MappingConfig()
    observations = [samples([[0, 0, 2], [0.02, 0, 2]]), samples([[0.01, 0, 2]])]
    references = [pose(x=0.2, yaw=0.02), pose(x=0.23, yaw=0.04)]
    maps = []
    for anchor in (pose(), pose(x=0.317, yaw=0.123)):
        rows = [
            row(i + 2, T=anchor @ pose(x=i * 0.03), reference=references[i])
            for i in range(2)
        ]
        maps.append(
            build_submap(
                [Keyframe(r, s) for r, s in zip(rows, observations, strict=True)],
                1,
                config,
            )
        )
    np.testing.assert_array_equal(maps[0].T_MW_reference, maps[1].T_MW_reference)
    for name in ("points", "keys", "frame_support", "position_std_m"):
        np.testing.assert_array_equal(
            getattr(maps[0].reference, name), getattr(maps[1].reference, name)
        )
