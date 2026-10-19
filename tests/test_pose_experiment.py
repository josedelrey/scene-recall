"""Source isolation, correction completion and frozen-population map tests."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from scene_recall.data import Calibration
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.pose_evaluation import evaluate_pose_maps, score_maps
from scene_recall.multiview.pose_runner import (
    interpolate_corrections,
    prepare_training_tracks,
    run_pose_experiment,
)
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.pipeline import TrajectoryFrame
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.run_metadata import sha256


def synthetic_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rgb = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    for i in range(7):
        Image.fromarray(rgb).save(source / f"rgb_{i}.png")
        Image.fromarray(np.full((480, 640), 10000, np.uint16)).save(
            source / f"depth_{i}.png"
        )
    for channel in ("rgb", "depth"):
        (source / f"{channel}.txt").write_text(
            "".join(f"{1 + i * 0.1:.1f} {channel}_{i}.png\n" for i in range(7))
        )
    (source / "groundtruth.txt").write_text(
        "".join(f"{1 + i * 0.1:.1f} 0 0 0 0 0 0 1\n" for i in range(7))
    )
    odometry = tmp_path / "odometry"
    run_odometry(
        source,
        odometry,
        dataset="tum_rgbd",
        backend_name="sparse",
        config=SparseConfig(max_features=300),
    )
    return source, odometry


def test_training_extraction_never_uses_heldout_images_depth_or_references(
    tmp_path, monkeypatch
):
    import scene_recall.multiview.pose_runner as runner

    _, odometry = synthetic_source(tmp_path)
    config = MappingConfig(keyframe_stride=2)
    first = prepare_training_tracks(odometry, config, 1)
    verified = runner.verified_observations
    extracted = runner.extract_rgbd_features
    calls = []

    def poisoned(*args):
        for observation in verified(*args):
            if observation.frame_index % 2:
                reference = np.eye(4)
                reference[:3, 3] = 1e6
                observation = replace(
                    observation,
                    rgb=np.zeros_like(observation.rgb),
                    depth=np.full_like(observation.depth, np.nan),
                    T_WC_D=reference,
                )
            yield observation

    def checked(observation, *args):
        assert observation.frame_index % 2 == 0
        calls.append(observation.frame_index)
        return extracted(observation, *args)

    monkeypatch.setattr(runner, "verified_observations", poisoned)
    monkeypatch.setattr(runner, "extract_rgbd_features", checked)
    second = prepare_training_tracks(odometry, config, 1)
    assert calls == [0, 2, 4, 6]
    for key, value in measurement_arrays(first[4]).items():
        np.testing.assert_array_equal(value, measurement_arrays(second[4])[key])


def test_interpolation_preserves_segment_roots_and_extrinsic_convention():
    K = np.eye(3)
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 0.12
    calibration = Calibration(K, K, extrinsic, (2, 2), (2, 2))
    rows = []
    for i in range(6):
        segment = i // 3
        pose = np.eye(4)
        pose[0, 3] = 0.1 * (i % 3)
        rows.append(
            TrajectoryFrame(
                i,
                None,
                segment,
                "initialized" if i == 0 else "lost" if i == 3 else "tracked",
                pose,
                None,
            )
        )
    training = {}
    for i in (0, 2, 3, 5):
        pose = rows[i].T_SC_D.copy()
        if i in (2, 5):
            pose[1, 3] = 0.04
        training[i] = pose @ np.linalg.inv(extrinsic)
    refined = interpolate_corrections(rows, training, calibration)
    np.testing.assert_array_equal(refined[0].T_SC_D, rows[0].T_SC_D)
    np.testing.assert_array_equal(refined[3].T_SC_D, rows[3].T_SC_D)
    assert refined[1].T_SC_D[1, 3] == pytest.approx(0.02)
    assert refined[4].T_SC_D[1, 3] == pytest.approx(0.02)
    assert refined[3].pair.T_C1C0 is None
    for i in (1, 2, 4, 5):
        np.testing.assert_allclose(
            refined[i].T_SC_D @ refined[i].pair.T_C1C0,
            refined[i - 1].T_SC_D,
            atol=1e-14,
        )


def test_coverage_loss_cannot_masquerade_as_better_map_geometry():
    observed = np.full((2, 2), 2.0)
    baseline = observed + 0.01
    partial = observed.copy()
    partial[0] = np.nan
    predictions = {
        f"{method}_{variant}_reference": image
        for method in ("voxel", "tsdf")
        for variant, image in (("sparse", baseline), ("anchored", partial))
    }
    result = score_maps(observed, predictions)
    baseline_metric = result["scores"]["voxel_sparse_reference"]["own"]
    refined_metric = result["scores"]["voxel_anchored_reference"]["own"]
    assert refined_metric["absolute_depth_error_m"]["mean"] == 0
    assert refined_metric["coverage"] == 0.5
    assert (
        refined_metric["missing_penalized_capped_mae_m"]["0.2"]
        > baseline_metric["missing_penalized_capped_mae_m"]["0.2"]
    )
    assert result["pairs"]["voxel_anchored_reference"]["sparse"]["target_samples"] == 2
    assert refined_metric["target_samples"] == baseline_metric["target_samples"] == 4


def test_experiment_roundtrip_preserves_baseline_and_frozen_map_populations(tmp_path):
    _, odometry = synthetic_source(tmp_path)
    original = {
        name: sha256(odometry / name)
        for name in ("trajectory.npz", "report.json", "frames.csv")
    }
    experiment = tmp_path / "pose"
    config = MappingConfig(keyframe_stride=2, pixel_stride=32, voxel_size_m=0.1)
    report = run_pose_experiment(odometry, experiment, mapping_config=config, offset=1)
    assert report["repeat_exact"] is True
    assert report["split"]["train_indices"] == [0, 2, 4, 6]
    saved, rows = load_trajectory(experiment / "anchored")
    assert saved["backend"] == "pose_only_anchored"
    assert len(rows) == 7
    assert saved["timing"]["pose_only_seconds"] >= 0
    maps = evaluate_pose_maps(experiment, tmp_path / "maps", variants=("anchored",))
    assert maps["split"] == report["split"]
    assert [r["frame_index"] for r in maps["views"]] == [1, 3, 5]
    assert all(sha256(odometry / name) == digest for name, digest in original.items())
    with pytest.raises(ValueError, match="exists"):
        run_pose_experiment(odometry, experiment)
    with pytest.raises(ValueError, match="chunks"):
        evaluate_pose_maps(
            experiment, tmp_path / "bad_chunks", variants=("anchored",), chunk_ids=[100]
        )
    # Refined archive alterations must be caught before scoring another cohort.
    with (experiment / "anchored/trajectory.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        evaluate_pose_maps(experiment, tmp_path / "changed", variants=("anchored",))
    assert not (tmp_path / "changed").exists()


def test_changed_heldout_source_is_verified_before_publication(tmp_path):
    source, odometry = synthetic_source(tmp_path)
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(source / "rgb_1.png")
    output = tmp_path / "pose"
    with pytest.raises(ValueError, match="fingerprint"):
        run_pose_experiment(
            odometry, output, mapping_config=MappingConfig(keyframe_stride=2), offset=1
        )
    assert not output.exists()


def test_failed_pose_publication_preserves_baseline_and_discards_staging(
    tmp_path, monkeypatch
):
    import scene_recall.multiview.pose_runner as runner

    _, odometry = synthetic_source(tmp_path)
    before = sha256(odometry / "trajectory.npz")

    def fail(*_args, **_kwargs):
        raise OSError("publication failure")

    monkeypatch.setattr(runner, "save_run", fail)
    output = tmp_path / "pose"
    with pytest.raises(OSError, match="publication failure"):
        run_pose_experiment(
            odometry, output, mapping_config=MappingConfig(keyframe_stride=2), offset=1
        )
    assert not output.exists()
    assert sha256(odometry / "trajectory.npz") == before
    assert not list(tmp_path.glob(".pose-only-*"))
