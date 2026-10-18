"""Training-only consensus and coverage accounting on controlled geometry."""

import runpy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from scene_recall.mapping.voxel import VoxelCloud

SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/support_aware_fusion.py"
)
CONSENSUS = SCRIPT["free_space_consensus"]
K = np.eye(3)


def cloud(points=((0, 0, 1),), support=(1,)):
    n = len(points)
    return VoxelCloud(
        np.arange(n * 3).reshape(n, 3),
        np.array(points, dtype=float).reshape(-1, 3),
        np.full((n, 3), 120, dtype=np.uint8),
        np.array(support),
        np.ones(n, dtype=np.int64),
        np.zeros(n),
    )


def views(depths):
    return [(i, np.eye(4), np.full((3, 3), z)) for i, z in enumerate(depths)]


@pytest.mark.parametrize(
    "depths,keep,agree,contradict",
    [
        ([1, 2], True, 1, 1),
        ([1, 2, 2], False, 1, 2),
        ([1, 1, 2, 2], True, 2, 2),
        ([1, 1, 2, 2, 2], False, 2, 3),
        ([0.5, 0.5, np.nan, 0, -1], True, 0, 0),
        ([1, 1.04, 0.96], True, 3, 0),
        ([], True, 0, 0),
    ],
)
def test_distinct_view_majority_ignores_occlusion_and_missing_depth(
    depths, keep, agree, contradict
):
    source = cloud()
    subset, evidence = CONSENSUS(source, views(depths), K, 0.05)
    assert evidence["keep"].tolist() == [keep]
    assert evidence["agreements"].tolist() == [agree]
    assert evidence["contradictions"].tolist() == [contradict]
    for f in fields(source):
        np.testing.assert_array_equal(
            getattr(subset, f.name), getattr(source, f.name)[[keep]]
        )
    assert source.points.tolist() == [[0, 0, 1]]


def test_supported_voxels_survive_even_unanimous_free_space_votes():
    source = cloud(((0, 0, 1), (1, 0, 1)), (1, 2))
    subset, evidence = CONSENSUS(source, views([2, 2, 2]), K, 0.05)
    assert evidence["keep"].tolist() == [False, True]
    assert subset.frame_support.tolist() == [2]


def test_inverse_camera_pose_native_pixel_and_behind_camera_handling():
    source = cloud(((1, 0, 2), (1, 0, 0.5), (10, 0, 2)), (1, 1, 1))
    pose = np.eye(4)
    pose[:3, 3] = [1, 0, 1]
    depth = np.full((3, 3), np.nan)
    depth[0, 0] = 2
    # Correct inverse transform gives (0, 0, 1) at native pixel (0, 0).
    _, evidence = CONSENSUS(source, [(i, pose, depth) for i in range(2)], K, 0.05)
    assert evidence["keep"].tolist() == [False, True, True]
    angle = np.pi / 2
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    pose[:3, :3] = rotation
    source = cloud((rotation @ [0, 0, 1] + pose[:3, 3],))
    _, evidence = CONSENSUS(source, [(i, pose, depth) for i in range(2)], K, 0.05)
    assert evidence["keep"].tolist() == [False]


def test_duplicate_frames_and_invalid_views_cannot_create_evidence():
    with pytest.raises(ValueError, match="distinct"):
        CONSENSUS(cloud(), [(0, np.eye(4), np.ones((3, 3)))] * 2, K, 0.05)
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        CONSENSUS(cloud(), [(0, pose, np.ones((3, 3)))], K, 0.05)
    with pytest.raises(ValueError, match="positive"):
        CONSENSUS(cloud(), [], K, np.nan)


def test_empty_cloud_is_valid():
    subset, evidence = CONSENSUS(cloud((), ()), views([1, 2]), K, 0.05)
    assert subset.points.shape == (0, 3)
    assert evidence["keep"].size == 0


def test_dropping_hard_geometry_cannot_hide_coverage_loss():
    observed = np.ones(4)
    baseline = np.array([1, 2, 1.02, np.nan])
    proposal = np.array([1, np.nan, np.nan, np.nan])
    anchor = np.array([True, True, True, False])
    score = SCRIPT["compare_predictions"](observed, baseline, proposal, anchor)
    paired = score["paired_baseline_common"]
    assert paired["baseline"]["target_samples"] == 1
    assert paired["consensus"]["absolute_depth_error_m"]["mean"] == 0
    before, after = (score["all_valid"][k] for k in ("baseline", "consensus"))
    assert before["coverage"] == 0.75
    assert after["coverage"] == 0.25
    assert before["thresholds"]["0.05"]["explained_fraction_of_target"] == 0.5
    assert after["thresholds"]["0.05"]["explained_fraction_of_target"] == 0.25
    assert (
        after["missing_penalized_capped_mae_m"]["0.2"]
        > before["missing_penalized_capped_mae_m"]["0.2"]
    )
    assert score["transitions"]["lost_previously_within_5cm"] == 1
    assert score["baseline_common"]["consensus"]["missing_samples"] == 2


def test_empty_evaluation_population_remains_unavailable():
    result = SCRIPT["coverage_score"](np.ones(1), np.ones(1), np.zeros(1, dtype=bool))
    assert result["coverage"] is None
    assert result["missing_penalized_capped_mae_m"]["0.2"] is None


def test_runner_repeats_without_target_leakage_and_rejects_changed_inputs(tmp_path):
    from PIL import Image
    from test_heldout import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    outputs = []
    for index, depth in enumerate((2.0, 2.3)):
        folder = tmp_path / str(index)
        folder.mkdir()
        source, odometry = continuous_run(folder, heldout_depth=depth)
        baseline, output = folder / "baseline", folder / "consensus"
        run_heldout(
            [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
        )
        result = SCRIPT["run"](baseline, [odometry], output)
        assert result["split"]["train_indices"] == [0, 2]
        assert len(result["views"]) == 1
        outputs.append(output)
        if index == 0:
            repeat = folder / "repeat"
            SCRIPT["run"](baseline, [odometry], repeat)
            verifier = runpy.run_path(
                Path(__file__).parents[1] / "experiments/verify_mapping_repeats.py"
            )["verify"]
            assert verifier(output, repeat)["status"] == "exact_match"
            # Keep conditional errors identical but change coverage accounting.
            import json

            report = json.loads((repeat / "report.json").read_text())
            report["pooled"]["gt"]["all_valid"]["consensus"]["coverage"] = 0
            (repeat / "report.json").write_text(json.dumps(report))
            with pytest.raises(ValueError, match="pooled"):
                verifier(output, repeat)
        Image.fromarray(np.full((480, 640), 12000, dtype=np.uint16)).save(
            source / "heldout.png"
        )
        with pytest.raises(AssertionError):
            SCRIPT["run"](baseline, [odometry], folder / "corrupt")
        assert not (folder / "corrupt").exists()
        assert not list(folder.glob(".consensus-*"))
    for label in ("sparse", "gt"):
        for suffix in ("consensus", "evidence"):
            name = f"{label}_000_{suffix}.npz"
            with np.load(outputs[0] / name) as a, np.load(outputs[1] / name) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_unpaired_chunk_keeps_geometry_and_withholds_scores(tmp_path):
    from test_heldout import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    _, odometry = continuous_run(tmp_path, missing_reference=0)
    baseline, output = tmp_path / "baseline", tmp_path / "consensus"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    result = SCRIPT["run"](baseline, [odometry], output)
    assert result["pooled"] == {}
    assert result["views"] == []
    assert result["unavailable_views"][0]["status"] == (
        "unavailable_unpaired_training_references"
    )
    with np.load(output / "sparse_000_consensus.npz") as data:
        assert len(data["points"]) > 0
