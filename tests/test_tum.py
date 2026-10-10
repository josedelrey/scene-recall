"""Analytic TUM fixtures for synchronization, units, frames, and strict parsing."""

import numpy as np
import pytest
from PIL import Image

from scene_recall.datasets.tum import (
    TUMConfig,
    TUMReader,
    associate_timestamps,
    load_tum_sequence,
)
from scene_recall.geometry.camera import backproject_depth
from scene_recall.odometry.pipeline import is_rigid


@pytest.fixture
def tum_sequence(tmp_path):
    for folder in ("rgb", "depth"):
        (tmp_path / folder).mkdir()
    rgb = np.empty((480, 640, 3), dtype=np.uint8)
    rgb[:] = [17, 101, 239]
    depth = np.full((480, 640), 5000, dtype=np.uint16)
    depth[0, :3] = [0, 10000, 65535]
    for index in range(3):
        Image.fromarray(rgb).save(tmp_path / "rgb" / f"{index}.png")
        Image.fromarray(depth).save(tmp_path / "depth" / f"{index}.png")
    (tmp_path / "rgb.txt").write_text(
        "# RGB\n0.51 rgb/0.png\n1.01 rgb/1.png\n1.51 rgb/2.png\n"
    )
    (tmp_path / "depth.txt").write_text(
        "# Depth\n0.5 depth/0.png\n1.0 depth/1.png\n1.5 depth/2.png\n"
    )
    # A 180 degree rotation about z with linear x translation.
    (tmp_path / "groundtruth.txt").write_text(
        "0 0 0 0 0 0 0 1\n1 2 0 0 0 0 1 0\n2 4 0 0 0 0 0 -1\n"
    )
    return tmp_path


def test_canonical_units_rgb_calibration_and_registered_frame(tum_sequence):
    sequence = load_tum_sequence(tum_sequence)
    calibration = sequence.calibration
    np.testing.assert_array_equal(calibration.K_D, calibration.K_R)
    np.testing.assert_array_equal(calibration.T_RD, np.eye(4))
    first = sequence.observations[0]
    assert first.rgb.dtype == np.uint8
    np.testing.assert_array_equal(first.rgb[0, 0], [17, 101, 239])
    assert first.depth.dtype == np.float32
    assert np.isnan(first.depth[0, 0])
    assert first.depth[0, 1] == 2
    assert first.depth[0, 2] == pytest.approx(65535 / 5000)
    assert first.depth[1, 1] == 1
    assert first.timestamp == 0.5
    # Default GT tolerance cannot support this deliberately sparse GT fixture.
    assert first.T_WC_D is None
    assert first.provenance["rgb_depth_difference_s"] == pytest.approx(0.01)
    points = backproject_depth(first.depth, calibration.K_D)
    np.testing.assert_allclose(points[240, 320], [0.5 / 525, 0.5 / 525, 1])


def test_ground_truth_slerp_direction_and_depth_timestamp(tum_sequence):
    config = TUMConfig(max_ground_truth_difference_s=0.51)
    sequence = load_tum_sequence(tum_sequence, config=config)
    first, exact, last = sequence.observations
    expected = np.eye(4)
    expected[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected[0, 3] = 1
    np.testing.assert_allclose(first.T_WC_D, expected, atol=1e-14)
    assert first.provenance["ground_truth_weight"] == 0.5
    assert exact.T_WC_D[0, 3] == 2
    assert last.T_WC_D[0, 3] == 3
    assert all(is_rigid(o.T_WC_D) for o in sequence.observations)
    # Camera -> world rotates +x toward +y, then translates +x.
    np.testing.assert_allclose((first.T_WC_D @ [1, 0, 0, 1])[:3], [1, 1, 0], atol=1e-14)
    # Opposite quaternion signs describing identical orientations interpolate safely.
    (tum_sequence / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n1 2 0 0 0 0 0 -1\n")
    first = load_tum_sequence(tum_sequence, config=config).observations[0]
    np.testing.assert_array_equal(first.T_WC_D[:3, :3], np.eye(3))


def test_no_extrapolation_and_missing_ground_truth_keeps_frames(tum_sequence):
    (tum_sequence / "groundtruth.txt").write_text(
        "0.75 0 0 0 0 0 0 1\n1.25 0 0 0 0 0 0 1\n"
    )
    sequence = load_tum_sequence(
        tum_sequence, config=TUMConfig(max_ground_truth_difference_s=0.3)
    )
    assert [o.T_WC_D is None for o in sequence.observations] == [True, False, True]
    (tum_sequence / "groundtruth.txt").unlink()
    assert all(o.T_WC_D is None for o in load_tum_sequence(tum_sequence).observations)


def test_selection_rebases_associated_indices_and_closes(tum_sequence):
    with TUMReader(tum_sequence, start=1, stop=3) as reader:
        first = next(reader)
        assert first.frame_index == 0
        assert first.provenance["original_frame_index"] == 1
        assert first.provenance["rgb_source_index"] == 1
        assert not reader.closed
    assert reader.closed
    assert list(reader) == []
    with pytest.raises(ValueError, match="closed"):
        reader.__enter__()
    empty = load_tum_sequence(tum_sequence, start=3)
    assert empty.observations == ()
    complete = TUMReader(tum_sequence)
    assert len(list(complete)) == 3
    assert complete.closed


def test_association_is_unique_global_greedy_and_deterministic():
    assert associate_timestamps([0, 0.125], [0.0625], 0.125) == [(0, 0)]
    assert associate_timestamps([0, 0.125], [0.1, 0.25], 0.125) == [(1, 0)]
    assert associate_timestamps([0, 1], [0.01, 5], 0.02) == [(0, 0)]
    assert associate_timestamps([0], [0.125], 0.125) == [(0, 0)]
    assert associate_timestamps([], [], 0.02) == []


def test_unmatched_counts_and_association_before_range_selection(tum_sequence):
    (tum_sequence / "rgb.txt").write_text(
        "0.51 rgb/0.png\n0.75 rgb/1.png\n1.01 rgb/1.png\n1.51 rgb/2.png\n"
    )
    reader = TUMReader(tum_sequence, start=1, stop=2)
    assert reader.metadata["unmatched_rgb_frames"] == 1
    observation = next(reader)
    assert observation.provenance["rgb_source_index"] == 2
    assert observation.provenance["depth_source_index"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "nan rgb/0.png",
        "1 rgb/0.png\n1 rgb/1.png",
        "2 rgb/0.png\n1 rgb/1.png",
        "0",
        "0 rgb/0.png extra",
        "bad rgb/0.png",
        "0 ../outside.png",
        "0 /tmp/outside.png",
    ],
)
def test_malformed_image_tables_rejected(tum_sequence, text):
    (tum_sequence / "rgb.txt").write_text(text)
    with pytest.raises(ValueError):
        TUMReader(tum_sequence)


@pytest.mark.parametrize(
    "text",
    [
        "0 0 0 0 0 0 0 0",
        "0 0 0 0 0 0 0 2",
        "0 nan 0 0 0 0 0 1",
        "0 0 0 0 0 0 1",
        "0 0 0 0 0 0 0 1\n0 0 0 0 0 0 0 1",
    ],
)
def test_malformed_ground_truth_rejected(tum_sequence, text):
    (tum_sequence / "groundtruth.txt").write_text(text)
    with pytest.raises(ValueError):
        TUMReader(tum_sequence)


def test_quaternion_rounding_is_normalized(tum_sequence):
    (tum_sequence / "groundtruth.txt").write_text("0.5 0 0 0 0 0 0 0.99999")
    pose = next(TUMReader(tum_sequence)).T_WC_D
    np.testing.assert_array_equal(pose, np.eye(4))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"start": -1}, IndexError),
        ({"stop": 4}, IndexError),
        ({"start": 2, "stop": 1}, IndexError),
        ({"start": True}, TypeError),
        ({"stop": 1.0}, TypeError),
    ],
)
def test_invalid_selection(tum_sequence, kwargs, error):
    with pytest.raises(error):
        TUMReader(tum_sequence, **kwargs)


@pytest.mark.parametrize(
    "kind", ["rgb_shape", "depth_shape", "depth_dtype", "rgb_mode", "missing"]
)
def test_malformed_images_close_reader(tum_sequence, kind):
    if kind == "rgb_shape":
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
            tum_sequence / "rgb/0.png"
        )
    elif kind == "depth_shape":
        Image.fromarray(np.zeros((10, 10), dtype=np.uint16)).save(
            tum_sequence / "depth/0.png"
        )
    elif kind == "depth_dtype":
        Image.fromarray(np.ones((480, 640), dtype=np.uint8)).save(
            tum_sequence / "depth/0.png"
        )
    elif kind == "rgb_mode":
        Image.fromarray(np.ones((480, 640), dtype=np.uint8)).save(
            tum_sequence / "rgb/0.png"
        )
    else:
        (tum_sequence / "depth/0.png").unlink()
    reader = TUMReader(tum_sequence)
    with pytest.raises((ValueError, OSError)):
        next(reader)
    assert reader.closed


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_association_limits(value):
    with pytest.raises(ValueError):
        TUMConfig(max_rgb_depth_difference_s=value)
    with pytest.raises(ValueError):
        TUMConfig(max_ground_truth_difference_s=value)


def test_ground_truth_gap_requires_both_bracket_samples(tum_sequence):
    # A nearby sample alone cannot justify interpolation across a large gap.
    (tum_sequence / "groundtruth.txt").write_text(
        "0.4 0 0 0 0 0 0 1\n0.51 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n1.5 0 0 0 0 0 0 1\n"
    )
    sequence = load_tum_sequence(tum_sequence)
    assert len(sequence.observations) == 3
    first, second, third = sequence.observations
    assert first.T_WC_D is None
    assert (
        first.provenance["pose_unavailable_reason"]
        == "ground truth bracket exceeds tolerance"
    )
    assert second.T_WC_D is not None
    assert third.T_WC_D is not None
