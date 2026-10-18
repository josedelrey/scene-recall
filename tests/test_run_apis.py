"""Exercise reusable run APIs without command-line modules or the optional SDK."""

import subprocess
import sys

import pytest
from test_mapping_cli import make_run

from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig


def test_programmatic_tracking_replay_and_mapping_without_cli_modules(tmp_path):
    source, _ = make_run(tmp_path)
    code = """
import hashlib
import json
from pathlib import Path
import sys

for name in (
    'scene_recall.odometry.cli',
    'scene_recall.mapping.cli',
    'scene_recall.visualization.cli',
    'rerun',
):
    sys.modules[name] = None

from scene_recall.datasets.replay import (
    open_source, validate_source, verified_observations,
)
from scene_recall.mapping.heldout import run_heldout
from scene_recall.mapping.runner import run_mapping
from scene_recall.mapping.voxel import MappingConfig
from scene_recall.odometry.io import load_trajectory
from scene_recall.odometry.runner import run_odometry
from scene_recall.odometry.sparse import SparseConfig
from scene_recall.visualization.recording import export_recording

source, output = map(Path, sys.argv[1:])
report = run_odometry(
    source, output, dataset='tum_rgbd', backend_name='sparse', config=SparseConfig(),
)
assert report['environment']['invocation'] == []
assert report['evaluation']['frame_count'] == 3
assert report['evaluation']['failed_transitions'] == 2
saved, frames = load_trajectory(output)
assert saved == json.loads(json.dumps(report))
assert [row.segment_id for row in frames] == [0, 1, 2]
digest, metadata = hashlib.sha256(), {}
reader, _ = open_source(saved)
with reader:
    validate_source(reader, saved)
    observations = list(verified_observations(reader, saved, frames, digest, metadata))
assert len(observations) == len(metadata) == 3
assert digest.hexdigest() == saved['source']['canonical_input_sha256']
mapping = run_mapping(
    output, output.parent / 'mapping', MappingConfig(pixel_stride=32, keyframe_stride=1),
)
assert mapping['frame_count'] == mapping['keyframe_count'] == 3
assert mapping['segment_count'] == len(mapping['submaps']) == 3
"""
    subprocess.run(
        [sys.executable, "-c", code, str(source), str(tmp_path / "programmatic")],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "dataset, backend, error, message",
    [
        ("unknown", "sparse", ValueError, "dataset"),
        ("tum_rgbd", "unknown", ValueError, "backend"),
        ("tum_rgbd", "icp", TypeError, "configuration"),
    ],
)
def test_odometry_runner_rejects_incompatible_selection_before_io(
    tmp_path, dataset, backend, error, message
):
    output = tmp_path / "run"
    with pytest.raises(error, match=message):
        run_odometry(
            tmp_path / "missing-source",
            output,
            dataset=dataset,
            backend_name=backend,
            config=SparseConfig(),
        )
    assert not output.exists()
