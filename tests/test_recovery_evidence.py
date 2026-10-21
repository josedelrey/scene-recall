"""Replay reports must distinguish new measurements from input-run diagnostics."""

from copy import deepcopy
from pathlib import Path


def test_replay_headers_preserve_science_and_label_inherited_summaries(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    from slam_recovery import execution_header

    original = {
        "initial_pose_stage": "ransac",
        "environment": {"package_sha256": "input"},
        "timing": {"wall_seconds": 999},
        "measurement_consistency": {"factor_count": 999},
        "frames": [{"status": "lost", "diagnostics": {"current": 7}}],
        "evaluation": {"failed_transitions": 1},
    }
    before = deepcopy(original)
    result = execution_header(
        original,
        {"package_sha256": "replay", "driver_sha256": "driver"},
        {"mean": 0.2},
    )
    assert original == before
    assert result["frames"] == before["frames"]
    assert result["evaluation"] == before["evaluation"]
    assert result["environment"]["package_sha256"] == "replay"
    assert result["timing"]["process_and_atlas_seconds"]["mean"] == 0.2
    assert "measurement_consistency" not in result
    assert result["input_report_header_context"]["measurement_consistency"] == {
        "factor_count": 999
    }
    assert result["initial_pose_stage"] is None
