"""Raw-factor ownership, joint bridge geometry and reversible map membership."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np

from scene_recall.data import Observation
from scene_recall.features import RGBDFeatures
from scene_recall.multiview.fixed_landmarks import make_direct_factors
from scene_recall.multiview.pose_only import PoseOnlyConfig
from scene_recall.slam.observation_welding import ObservationWelding
from scene_recall.slam.pose_graph import MetricPoseGraph
from scene_recall.slam.system import IncrementalSLAM, SLAMConfig

scene = runpy.run_path(
    str(Path(__file__).parents[1] / "experiments/temporal_controls.py")
)["scene"]


def bridge():
    calibration, tracks, samples, truth = scene(5, exact=False, skew=0)
    factors, _ = make_direct_factors(
        tracks,
        samples,
        list(range(5)),
        calibration,
        PoseOnlyConfig(min_grid_coverage=0),
    )
    descriptors = np.random.default_rng(4).normal(size=(96, 128)).astype(np.float32)
    features = [
        RGBDFeatures(frame.features.pixels, descriptors.copy(), frame.features.points_R)
        for frame in tracks.frames
    ]
    graph = MetricPoseGraph()
    weld = ObservationWelding(calibration)
    for frame in range(5):
        segment = 0 if frame < 3 else 1
        local = tracks.frames[frame].T_SR @ samples[frame].T_RcD
        if segment:
            local = np.linalg.inv(tracks.frames[3].T_SR) @ local
        graph.add_node(frame, segment, local)
        if frame not in (0, 3):
            graph.add_edge(
                frame - 1,
                frame,
                np.linalg.inv(graph.local[frame]) @ graph.local[frame - 1],
            )
        mask = factors.view == frame
        packet = replace(
            factors,
            **{
                name: getattr(factors, name)[mask]
                for name in factors.__dataclass_fields__
            },
        )
        if segment:
            packet = replace(packet, track_id=packet.track_id + 1000)
        measurement = {
            **{name: getattr(packet, name) for name in packet.__dataclass_fields__},
            "feature_index": factors.track_id[mask],
            "native_index": factors.track_id[mask],
            "frontend_identity": factors.track_id[mask],
        }
        weld.admit(frame, segment, measurement)
    transform = np.linalg.inv(truth[4]) @ truth[0]
    return graph, weld, features, transform


def test_verified_bridge_joins_raw_tracks_and_revises_joint_state():
    graph, weld, features, transform = bridge()
    association = weld.propose(0, 4, features[0], features[4], transform, 0.75)
    assert association["status"] == "admitted"
    assert association["associations"] >= 20
    graph.add_edge(0, 4, transform, kind="loop", group=(0, 4))
    graph.solve()
    report = weld.refine(graph)
    assert len(report) == 1
    assert report[0]["observations"] > 100
    assert report[0]["landmarks"] >= 20
    assert report[0]["status"] in ("refined", "unchanged")
    identities, accepted = weld._identities({(0, 4)})
    assert accepted[0, 4] >= 20
    assert any(
        identities[weld.identity_map[0, i]] == identities[weld.identity_map[1, i]]
        for i in range(96)
    )


def test_revocation_removes_cross_map_identity_without_mutating_native_packets():
    graph, weld, features, transform = bridge()
    weld.propose(0, 4, features[0], features[4], transform, 0.75)
    fingerprints = {
        frame: view.factors.fingerprint() for frame, view in weld.views.items()
    }
    graph.add_edge(0, 4, transform, kind="loop", group=(0, 4))
    graph.solve()
    weld.refine(graph)
    graph.revoke(0, 4, "controlled bridge removal")
    graph.solve()
    assert weld.refine(graph) == []
    identities, _ = weld._identities(set())
    assert all(
        identities[weld.identity_map[0, i]] != identities[weld.identity_map[1, i]]
        for i in range(96)
    )
    assert len(set(graph.components.values())) == 2
    assert fingerprints == {
        frame: view.factors.fingerprint() for frame, view in weld.views.items()
    }


def test_raw_ledger_survives_local_landmark_budget(monkeypatch):
    from scene_recall.odometry.pipeline import PairEstimate

    calibration, tracks, samples, _ = scene(2, exact=True, skew=0)
    features = {
        i: RGBDFeatures(frame.features.pixels[:20], None, frame.features.points_R[:20])
        for i, frame in enumerate(tracks.frames)
    }
    samples = {
        i: replace(
            sample,
            **{
                name: getattr(sample, name)[:20]
                for name in ("points_D", "points_R", "source_index", "whitening_D")
            },
        )
        for i, sample in samples.items()
    }

    class Frontend:
        def start(self, observation, calibration):
            i = observation.frame_index
            self._pose = np.eye(4)
            self._previous = observation, features[i]
            self.keyframes = {i: True}
            self.owners = {i: np.arange(20, dtype=int) + 20 * i}

        def estimate(self, source, target, calibration):
            self.start(target, calibration)
            return PairEstimate(np.eye(4))

    engine = IncrementalSLAM(
        SLAMConfig(
            max_landmarks=20,
            keyframe_interval=1,
            optimize=False,
            exposure="static",
            raw_ledger=True,
        )
    )
    engine.frontend = Frontend()
    monkeypatch.setattr(
        "scene_recall.slam.system.bind_depth",
        lambda pixels, depth, *_: samples[int(depth[0, 0]) - 1],
    )
    counts = []
    for i in range(2):
        observation = Observation(
            i,
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.full((240, 320), i + 1, dtype=np.float32),
        )
        _, _, _, diagnostic = engine.process(observation, calibration)
        counts.append(diagnostic.get("measurement_count", 0))
        assert len(engine.last_raw_measurement["view"]) == 20
    assert counts == [20, 0]
    assert engine.last_measurement is None


def test_withheld_contradiction_rolls_back_joint_pose_revision(monkeypatch):
    graph, weld, features, transform = bridge()
    weld.propose(0, 4, features[0], features[4], transform, 0.75)
    graph.add_edge(0, 4, transform, kind="loop", group=(0, 4))
    graph.solve()
    original = {frame: pose.copy() for frame, pose in graph.poses.items()}
    version = graph.version

    def contradictory(poses, factors, K, keys, root, *, config):
        revised = poses.copy()
        revised[keys.index(4), 0, 3] += 0.2
        _, first = np.unique(factors.track_id, return_index=True)
        return (
            revised,
            factors.points[first],
            {
                "status": "refined",
                "reason": "controlled contradiction",
            },
        )

    monkeypatch.setattr(
        "scene_recall.slam.observation_welding.solve_window", contradictory
    )
    report = weld.refine(graph)
    assert report[0]["status"] == "validation_rejected"
    assert graph.version == version
    for frame, pose in original.items():
        np.testing.assert_array_equal(graph.poses[frame], pose)
