"""Protect matched-population diagnostics from losses and unavailable references."""

import runpy
from pathlib import Path

import numpy as np

SCRIPT = runpy.run_path(
    Path(__file__).parents[1] / "experiments/analyze_odometry_sequences.py"
)
matched_errors = SCRIPT["matched_errors"]
summarize_pairs = SCRIPT["summarize_pairs"]
timestamp_diagnostic = SCRIPT["timestamp_diagnostic"]


def test_matching_excludes_icp_losses_without_hiding_sparse_hybrid_population():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    references = poses.copy()
    references[4] = np.nan
    archives = {
        name: {
            "segment_id": np.array([0, 0, 0, 0, 0]),
            "T_SC_D": poses.copy(),
            "T_WC_D_reference": references.copy(),
        }
        for name in ("sparse", "icp", "hybrid")
    }
    archives["icp"]["segment_id"] = np.array([0, 0, 1, 1, 1])
    archives["hybrid"]["T_SC_D"][3, 0, 3] = 0.1
    candidates = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    indices, errors = matched_errors(archives, candidates)
    assert indices == [[0, 1], [2, 3]]
    assert errors["sparse"] == [(0.0, 0.0), (0.0, 0.0)]
    np.testing.assert_allclose(errors["hybrid"], [(0, 0), (0.1, 0)])
    indices, errors = matched_errors(archives, candidates, ("sparse", "hybrid"))
    assert indices == [[0, 1], [1, 2], [2, 3], [0, 3]]
    assert len(errors["sparse"]) == len(errors["hybrid"]) == 4


def test_failure_summary_keeps_attempts_without_reference_errors():
    row = {
        name: {
            "status": "lost" if name == "icp" else "tracked",
            "error_m_deg": None,
            "candidate_error_m_deg": [0.1, 2.0] if name == "icp" else None,
            "diagnostics": {},
        }
        for name in ("sparse", "icp", "hybrid")
    }
    summary = summarize_pairs([row])
    assert summary["attempted"] == 1
    assert summary["icp"]["lost"] == 1
    assert summary["sparse"]["accepted_errors"]["translation_m"]["count"] == 0
    assert summary["icp"]["lost_candidate_errors"]["translation_m"]["rmse"] == 0.1
    assert summary["icp"]["lost_candidate_errors"]["rotation_deg"]["rmse"] == 2.0
    assert summary["icp"]["lost_candidate_errors"]["rotation_deg"]["count"] == 1


def test_clock_sensitivity_preserves_primary_estimates_and_endpoint_direction(tmp_path):
    from scene_recall.datasets.tum import TUMReader

    (tmp_path / "rgb.txt").write_text("1.01 rgb.png\n1.049 rgb.png\n")
    (tmp_path / "depth.txt").write_text("1 depth.png\n1.03 depth.png\n")
    ground_truth = []
    for stamp in (1.0, 1.03, 1.06):
        angle = np.radians(600 * (stamp - 1))
        ground_truth.append(
            f"{stamp} {stamp - 1} 0 0 0 0 {np.sin(angle / 2)} {np.cos(angle / 2)}"
        )
    (tmp_path / "groundtruth.txt").write_text("\n".join(ground_truth) + "\n")
    with TUMReader(tmp_path) as reader:
        references = np.array([reader._reference(stamp)[0] for stamp in (1, 1.03)])
    # Use exact depth-time motion. The second RGB exposure is 19 ms later.
    arrays = {
        "segment_id": np.array([0, 0]),
        "T_WC_D_reference": references.copy(),
        "T_SC_D": np.linalg.inv(references[0]) @ references,
        "T_C1C0": np.array(
            [np.full((4, 4), np.nan), np.linalg.inv(references[1]) @ references[0]]
        ),
    }
    report = {
        "source": {
            "dataset": "tum_rgbd",
            "path": str(tmp_path),
            "association_config": {
                "max_rgb_depth_difference_s": 0.02,
                "max_ground_truth_difference_s": 0.05,
            },
        },
        "frames": [
            {"provenance": {"timestamp_color": stamp}} for stamp in (1.01, 1.049)
        ],
    }
    diagnostic = timestamp_diagnostic(
        report, {name: arrays for name in ("sparse", "icp", "hybrid")}
    )
    metrics = diagnostic["accepted_pose_errors_same_population"]["sparse"]
    np.testing.assert_allclose(
        metrics["depth_time"]["rotation_deg"]["rmse"], 0, atol=1e-12
    )
    np.testing.assert_allclose(
        metrics["rgb_time"]["rotation_deg"]["rmse"], 5.4, atol=1e-10
    )
    endpoint = diagnostic["segment_endpoint_sensitivity"]["sparse"][0]
    np.testing.assert_allclose(endpoint["depth_time_error_m_deg"], [0, 0], atol=1e-12)
    np.testing.assert_allclose(endpoint["rgb_time_error_m_deg"][1], 5.4, atol=1e-10)
    np.testing.assert_array_equal(arrays["T_WC_D_reference"], references)


def test_historical_scannet_reports_have_no_clock_diagnostic():
    assert timestamp_diagnostic({"source": {"sequence_id": "scene0000_00"}}, {}) is None
