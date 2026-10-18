"""Budget, gauge, split isolation and reproducibility for motion keyframes."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers.mapping import pose, row

from scene_recall.mapping.keyframes import select_motion_keyframes

SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/compare_keyframe_selection.py"
)


def test_translation_budget_tracks_travel_instead_of_frame_density():
    positions = [0, 0.1, 0.2, 0.3, 0.4, 1.5, 2.6, 3.7, 3.8, 3.9]
    candidates = [(i, pose(x=x)) for i, x in enumerate(positions)]
    selected, evidence = select_motion_keyframes(candidates, 4)
    assert selected == [0, 5, 6, 9]
    assert evidence["translation_travel_m"] == pytest.approx(3.9)
    assert evidence["rotation_travel_rad"] == 0
    assert not evidence["zero_motion_fallback"]


def test_pure_rotation_and_global_gauge_preserve_selection():
    angles = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.1, 1.6]
    candidates = [(i, pose(yaw=angle)) for i, angle in enumerate(angles)]
    selected, evidence = select_motion_keyframes(candidates, 3)
    assert selected == [0, 5, 7]
    assert evidence["translation_travel_m"] == 0
    assert evidence["rotation_travel_rad"] == pytest.approx(1.6)
    gauge = pose(x=20, z=-3, yaw=0.32)
    assert (
        select_motion_keyframes([(i, gauge @ T) for i, T in candidates], 3)[0]
        == selected
    )


def test_stationary_fallback_ties_and_saturated_budget_keep_distinct_endpoints():
    candidates = [(i * 2, np.eye(4)) for i in range(7)]
    selected, evidence = select_motion_keyframes(candidates, 4)
    assert selected == [0, 4, 8, 12]
    assert evidence["zero_motion_fallback"]
    assert select_motion_keyframes(candidates, 7)[0] == [i for i, _ in candidates]
    assert select_motion_keyframes(candidates[:1], 1)[0] == [0]
    # Quantile halfway between candidate positions chooses the earlier one.
    assert select_motion_keyframes(candidates[:4], 3)[0] == [0, 2, 6]


def test_large_motion_jump_still_fills_exact_budget_without_duplicates():
    candidates = [(i, pose(x=100 if i >= 5 else 0)) for i in range(10)]
    selected, _ = select_motion_keyframes(candidates, 8)
    assert len(selected) == len(set(selected)) == 8
    assert selected == sorted(selected)
    assert selected[0] == 0 and selected[-1] == 9


@pytest.mark.parametrize("budget", [0, 1, 5, True, 2.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        select_motion_keyframes([(i, np.eye(4)) for i in range(4)], budget)


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf])
def test_invalid_rotation_scale_is_rejected(radius):
    with pytest.raises(ValueError, match="radius"):
        select_motion_keyframes([(0, np.eye(4))], 1, radius)


def test_candidate_order_and_rigidity_are_validated():
    with pytest.raises(ValueError, match="increasing"):
        select_motion_keyframes([(0, np.eye(4)), (0, np.eye(4))], 2)
    T = np.eye(4)
    T[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        select_motion_keyframes([(0, T)], 1)


def test_plan_never_reads_heldout_poses_or_references_and_respects_chunks():
    frames = [row(i, T=pose(x=i * i / 100)) for i in range(10)]
    split = {
        "chunks": [[0, 2, 4], [6, 9]],
        "target_indices_by_chunk": [[1, 3], [7]],
        "excluded_outside_chunk_extent": [5],
    }
    before = SCRIPT["selection_plan"](frames, split, 1)
    altered = [
        replace(
            f,
            reference_pose=pose(x=100 + i),
            T_SC_D=pose(x=1000) if i in (1, 3, 5, 7) else f.T_SC_D,
        )
        for i, f in enumerate(frames)
    ]
    assert before == SCRIPT["selection_plan"](altered, split, 1)
    for j, group in enumerate(before[0]):
        assert len(group["motion"]) == len(split["chunks"][j])
        assert not set(group["motion"]) & {1, 3, 5, 7}
    frames[8] = replace(frames[8], segment_id=1)
    with pytest.raises(ValueError, match="reset"):
        SCRIPT["selection_plan"](frames, split, 1)


def test_cohort_scoring_uses_exact_common_pixels_and_charges_missing_predictions():
    observed = np.ones(4)
    predictions = {
        "temporal": np.array([1.01, 2, np.nan, 1]),
        "motion": np.array([1.02, np.nan, 1, 1.01]),
    }
    anchor = np.array([True, True, False, False])
    scores, common = SCRIPT["score_cohort"](observed, predictions, anchor)
    assert common.tolist() == [True, False, False, True]
    for score in scores.values():
        assert score["common_all_versions"]["target_samples"] == 2
        assert score["all_valid"]["coverage"] == 0.75
        assert score["all_valid"]["missing_penalized_capped_mae_m"]["0.2"] >= 0.05
    assert scores["motion"]["baseline_common"]["missing_samples"] == 1


def test_runner_target_depth_independence_repeat_and_atomic_failure(tmp_path):
    from helpers.runs import continuous_run
    from PIL import Image

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    outputs = []
    for index, depth in enumerate((2.0, 2.3)):
        folder = tmp_path / str(index)
        folder.mkdir()
        source, odometry = continuous_run(folder, heldout_depth=depth)
        baseline, output = folder / "baseline", folder / "selection"
        run_heldout(
            [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
        )
        result = SCRIPT["run"](baseline, [odometry], output)
        assert result["selection"]["sparse"][0] == {
            "temporal": [0, 2],
            "motion": [0, 2],
        }
        assert result["views"][0]["status"] == "evaluated"
        outputs.append(output)
        if index == 0:
            repeat = folder / "repeat"
            SCRIPT["run"](baseline, [odometry], repeat)
            verify = runpy.run_path(
                Path(__file__).parents[1] / "experiments/verify_mapping_repeats.py"
            )["verify"]
            assert verify(output, repeat)["status"] == "exact_match"
            report = json.loads((repeat / "report.json").read_text())
            report["selection"]["sparse"][0]["motion"] = [0, 1]
            (repeat / "report.json").write_text(json.dumps(report))
            with pytest.raises(ValueError, match="selection"):
                verify(output, repeat)
        Image.fromarray(np.full((480, 640), 12000, dtype=np.uint16)).save(
            source / "heldout.png"
        )
        with pytest.raises(AssertionError):
            SCRIPT["run"](baseline, [odometry], folder / "corrupt")
        assert not (folder / "corrupt").exists()
        assert not list(folder.glob(".keyframes-*"))
    for strategy in ("temporal", "motion"):
        for mode in ("estimated", "reference"):
            name = f"sparse_{strategy}_000_{mode}.npz"
            with np.load(outputs[0] / name) as a, np.load(outputs[1] / name) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_missing_training_reference_preserves_maps_and_original_exclusion(tmp_path):
    from helpers.runs import continuous_run

    from scene_recall.mapping.heldout import run_heldout
    from scene_recall.mapping.voxel import MappingConfig

    _, odometry = continuous_run(tmp_path, missing_reference=0)
    baseline, output = tmp_path / "baseline", tmp_path / "selection"
    run_heldout(
        [odometry], baseline, MappingConfig(pixel_stride=32, keyframe_stride=2), 1
    )
    result = SCRIPT["run"](baseline, [odometry], output)
    assert result["pooled"] == {}
    assert result["views"][0]["status"] == "unavailable_unpaired_training_references"
    with np.load(output / "sparse_motion_000_estimated.npz") as archive:
        assert len(archive["points"]) > 0
