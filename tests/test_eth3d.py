"""Processed ETH3D image units, calibration and verified replay contracts."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from scene_recall.datasets.eth3d import ETH3DReader
from scene_recall.datasets.replay import (
    open_source,
    validate_source,
    verified_observations,
)
from scene_recall.datasets.tum import TUMConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig


@pytest.fixture
def eth3d_source(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    for i in range(3):
        image = np.full((24, 32), 17 + i, dtype=np.uint8)
        depth = np.full((24, 32), 10000, dtype=np.uint16)
        depth[0, 0] = 0
        Image.fromarray(image).save(tmp_path / "rgb" / f"{i}.png")
        Image.fromarray(depth).save(tmp_path / "depth" / f"{i}.png")
    (tmp_path / "rgb.txt").write_text(
        "".join(f"{i + 1}.0 rgb/{i}.png\n" for i in range(3))
    )
    (tmp_path / "depth.txt").write_text(
        "".join(f"{i + 1}.0 depth/{i}.png\n" for i in range(3))
    )
    (tmp_path / "groundtruth.txt").write_text(
        "".join(f"{i + 1}.0 {i / 10} 0 0 0 0 0 1\n" for i in range(3))
    )
    (tmp_path / "calibration.txt").write_text("100 110 15 12\n")
    return tmp_path


def test_explicit_intrinsics_non_tum_grid_units_and_gray_rgb(eth3d_source):
    with ETH3DReader(eth3d_source, start=1) as reader:
        rows = list(reader)
        assert reader.calibration.rgb_shape == (24, 32)
        np.testing.assert_array_equal(
            reader.calibration.K_R, [[100, 0, 15], [0, 110, 12], [0, 0, 1]]
        )
    assert len(rows) == 2
    assert rows[0].frame_index == 0
    assert rows[0].provenance["original_frame_index"] == 1
    np.testing.assert_array_equal(rows[0].rgb[0, 0], [18, 18, 18])
    assert rows[0].depth[1, 1] == 2
    assert np.isnan(rows[0].depth[0, 0])
    assert rows[0].T_WC_D[0, 3] == 0.1
    assert rows[0].provenance["timestamp_unit"] == "source_clock_seconds"


def test_eth3d_run_reopens_and_verifies_all_canonical_inputs(eth3d_source, tmp_path):
    output = tmp_path / "run"
    report = run_odometry(
        eth3d_source,
        output,
        dataset="eth3d_slam",
        backend_name="sparse",
        config=SparseConfig(max_features=20),
        stop=2,
    )
    _, rows = load_trajectory(output)
    reader, _ = open_source(report)
    digest = hashlib.sha256()
    with reader:
        validate_source(reader, report)
        actual = list(verified_observations(reader, report, rows, digest, {}))
    assert len(actual) == 2
    assert digest.hexdigest() == report["source"]["canonical_input_sha256"]


@pytest.mark.parametrize(
    "calibration", ["1 2 3\n", "0 110 15 12\n", "100 110 nan 12\n"]
)
def test_invalid_explicit_calibration_is_rejected(eth3d_source, calibration):
    (eth3d_source / "calibration.txt").write_text(calibration)
    with pytest.raises(ValueError):
        ETH3DReader(eth3d_source)


def test_unregistered_grid_is_rejected_without_guessing_a_scale(eth3d_source):
    Image.fromarray(np.ones((12, 16), dtype=np.uint16)).save(
        eth3d_source / "depth/0.png"
    )
    with pytest.raises(ValueError, match="equal registered grids"):
        ETH3DReader(eth3d_source)


def test_low_rate_reference_bracket_is_explicit_without_extrapolation(eth3d_source):
    stamps = [1, 1.074, 1.148]
    for name in ("rgb", "depth"):
        (eth3d_source / f"{name}.txt").write_text(
            "".join(f"{stamp:.9f} {name}/{i}.png\n" for i, stamp in enumerate(stamps))
        )
    (eth3d_source / "groundtruth.txt").write_text(
        "".join(
            f"{stamp + 0.00001:.9f} {i / 10} 0 0 0 0 0 1\n"
            for i, stamp in enumerate(stamps)
        )
    )
    with ETH3DReader(eth3d_source) as reader:
        strict = list(reader)
    with ETH3DReader(
        eth3d_source, config=TUMConfig(max_ground_truth_difference_s=0.1)
    ) as reader:
        bracketed = list(reader)
    assert all(row.T_WC_D is None for row in strict)
    assert bracketed[0].T_WC_D is None
    assert all(row.T_WC_D is not None for row in bracketed[1:])
    for a, b in zip(strict, bracketed, strict=True):
        np.testing.assert_array_equal(a.rgb, b.rgb)
        np.testing.assert_array_equal(a.depth, b.depth)
