"""Pose-independent RGB association with unchanged temporal RGB-D batch fitting."""

import argparse
import copy
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from scene_recall.mapping.voxel import MappingConfig
from scene_recall.multiview.batch_ba import BatchConfig
from scene_recall.multiview.batch_runner import fit_batch_packets, make_batch_packets
from scene_recall.multiview.io import measurement_arrays
from scene_recall.multiview.joint_runner import calibration_from_report, verify_packets
from scene_recall.multiview.temporal import TimingConfig
from scene_recall.multiview.temporal_runner import (
    export_color_poses,
    motion_table,
    prepare_temporal,
)
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.odometry.evaluation import evaluate_trajectory
from scene_recall.odometry.io import load_trajectory, save_run
from scene_recall.run_metadata import code_hash, sha256


def rgb_graph(graph, descriptors, offsets=(1, 2, 5, 10, 20)):
    """Mutual SIFT, robust fundamental matrices and unique-view identity merges.

    Neither supplied camera poses nor native depth enter match verification.
    Each pair uses the same seed, 1.5-pixel threshold, 0.999 confidence and 2000
    RANSAC trials. Pairs need at least 12 matches and 12 robust inliers.
    """
    ids = list(map(int, graph["frame_index"]))
    segments = dict(zip(ids, map(int, graph["segment_id"])))
    pixels = {
        i: graph["pixels"][
            graph["feature_offsets"][j] : graph["feature_offsets"][j + 1]
        ]
        for j, i in enumerate(ids)
    }
    proposals = []
    pairs = []
    matrices = []
    pair_counts = []
    for j, target in enumerate(ids):
        for offset in offsets:
            if j < offset:
                continue
            source = ids[j - offset]
            if segments[source] != segments[target]:
                continue
            matches, ratios = match_descriptors(
                descriptors[source], descriptors[target]
            )
            pairs.append((source, target))
            F = np.full((3, 3), np.nan)
            inlier = np.zeros(len(matches), bool)
            if len(matches) >= 12:
                cv2.setRNGSeed(0)
                estimate, mask = cv2.findFundamentalMat(
                    pixels[source][matches[:, 0]],
                    pixels[target][matches[:, 1]],
                    cv2.FM_RANSAC,
                    1.5,
                    0.999,
                    2000,
                )
                if (
                    estimate is not None
                    and estimate.shape == (3, 3)
                    and np.isfinite(estimate).all()
                    and mask is not None
                    and len(mask) == len(matches)
                ):
                    candidate = mask.ravel().astype(bool)
                    if candidate.sum() >= 12:
                        F = estimate
                        inlier = candidate
            matrices.append(F)
            pair_counts.append((len(matches), int(inlier.sum())))
            for (a, b), ratio in zip(matches[inlier], ratios[inlier], strict=True):
                proposals.append((float(ratio), (source, int(a)), (target, int(b))))
    owner = {}
    components = {}
    edges = []
    conflicts = 0
    for ratio, a, b in sorted(proposals):
        first, second = owner.get(a, a), owner.get(b, b)
        if first == second:
            edges.append((*a, *b, ratio, 2))
            continue
        one, two = components.get(first, (a,)), components.get(second, (b,))
        keys = tuple(sorted(one + two))
        if len({i for i, k in keys}) != len(keys):
            conflicts += 1
            edges.append((*a, *b, ratio, 0))
            continue
        root = keys[0]
        components.pop(first, None)
        components.pop(second, None)
        components[root] = keys
        owner.update({key: root for key in keys})
        edges.append((*a, *b, ratio, 1))
    tracks = [keys for _, keys in sorted(components.items())]
    result = {
        key: graph[key].copy()
        for key in (
            "frame_index",
            "segment_id",
            "timestamp_s",
            "T_SR",
            "feature_offsets",
            "pixels",
            "points_R",
        )
    }
    result.update(
        graph_model=np.array("rgb_fundamental_ransac_v1"),
        track_id=np.arange(len(tracks)),
        track_offsets=np.cumsum([0] + [len(t) for t in tracks]),
        observation_keys=np.array(
            [key for t in tracks for key in t], dtype=np.int64
        ).reshape(-1, 2),
        candidate_pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
        fundamental_matrices=np.array(matrices).reshape(-1, 3, 3),
        pair_match_inlier_counts=np.array(pair_counts, dtype=np.int64).reshape(-1, 2),
        accepted_edge_records=np.array(edges).reshape(-1, 6),
    )
    return result, {
        "candidate_pairs": len(pairs),
        "verified_pairs": int(
            np.isfinite(result["fundamental_matrices"]).all(axis=(1, 2)).sum()
        ),
        "descriptor_matches": sum(n for n, k in pair_counts),
        "pair_inliers": sum(k for n, k in pair_counts),
        "tracks": len(tracks),
        "observations": len(result["observation_keys"]),
        "duplicate_view_conflicts": conflicts,
    }


def _run(parent, output):
    if output.exists():
        raise ValueError("output already exists")
    report, protected = verify_packets(parent)
    output.mkdir(parents=True)
    frozen_code = code_hash()
    calibration = calibration_from_report(report)
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    with np.load(parent / "training_measurements.npz", allow_pickle=False) as a:
        old = {k: a[k] for k in a.files}
    with np.load(parent / "native_measurements.npz", allow_pickle=False) as a:
        native = {k: a[k] for k in a.files}
    if not (
        np.isfinite(native["color_timestamp_s"])
        & np.isfinite(native["depth_timestamp_s"])
    ).all():
        raise ValueError("RGB association experiment requires known acquisition clocks")
    started = perf_counter()
    prepared = prepare_temporal(
        Path(report["baseline_run"]),
        MappingConfig(**report["mapping_config"]),
        report["offset"],
        TimingConfig(**report["timing_config"]),
    )
    tracks, samples = prepared[4], prepared[5]
    for key, value in measurement_arrays(tracks).items():
        np.testing.assert_array_equal(value, old[key])
    for i, sample in samples.items():
        for field in sample.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(sample, field), native[f"frame_{i}_{field}"]
            )
    descriptors = {
        f.frame_index: np.empty((0, 128), dtype=np.float32)
        if f.features.descriptors is None
        else f.features.descriptors
        for f in tracks.frames
    }
    np.savez_compressed(
        output / "descriptors.npz", **{f"frame_{i}": d for i, d in descriptors.items()}
    )
    preparation_seconds = perf_counter() - started
    print(
        "raw packet equality verified",
        len(tracks.frames),
        "preparation seconds",
        preparation_seconds,
        flush=True,
    )
    started = perf_counter()
    graph, association = rgb_graph(old, descriptors)
    association_seconds = perf_counter() - started
    print("association", association, flush=True)
    config = BatchConfig()
    started = perf_counter()
    base, segments, packets = make_batch_packets(graph, native, calibration, config)
    factor_seconds = perf_counter() - started
    print(
        "packets",
        [
            (
                p[3]["segment_id"],
                len(p[0]),
                p[3]["candidate_tracks"],
                p[3]["root_has_observations"],
            )
            for p in packets
        ],
        flush=True,
    )
    started = perf_counter()
    poses, diagnostics, arrays = fit_batch_packets(
        base, segments, packets, calibration.K_R, config
    )
    solver_seconds = perf_counter() - started
    print(
        "fits",
        [
            (
                d["status"],
                d.get("reason"),
                d.get("iterations"),
                d.get("initial_cost"),
                d.get("final_cost"),
            )
            for d in diagnostics
        ],
        flush=True,
    )
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    refined = export_color_poses(rows, poses, calibration, motions)
    evaluation = evaluate_trajectory(
        refined,
        (1, 5, 10, 20, 50, 100, 200),
        time_intervals_s=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    child = copy.deepcopy(baseline)
    child.update(
        backend="experimental_rgb_geometry_batch",
        evaluation=evaluation,
        config=asdict(config),
    )
    child["environment"]["code_sha256"] = frozen_code
    np.savez_compressed(output / "training_measurements.npz", **graph)
    shutil.copyfile(
        parent / "native_measurements.npz", output / "native_measurements.npz"
    )
    for name in ("clock", "timed"):
        shutil.copytree(parent / name, output / name)
        shutil.copyfile(parent / f"{name}_factors.npz", output / f"{name}_factors.npz")
    np.savez_compressed(output / "rgb_factors.npz", **arrays)
    save_run(output / "rgb", refined, child, baseline["source"]["start"])
    result = copy.deepcopy(report)
    result.update(
        experiment="pose_independent_rgb_association_batch",
        parent_experiment=str(parent.resolve()),
        parent_report_sha256=sha256(parent / "report.json"),
        code_sha256=frozen_code,
        probe_sha256=sha256(Path(__file__)),
        raw_preparation_equal=True,
        config=asdict(config),
        rgb_association=dict(
            association,
            ratio=0.7,
            offsets=[1, 2, 5, 10, 20],
            threshold_px=1.5,
            confidence=0.999,
            max_iterations=2000,
            minimum_pair_support=12,
        ),
        cost_seconds={
            "raw_preparation": preparation_seconds,
            "association": association_seconds,
            "factors": factor_seconds,
            "solver": solver_seconds,
        },
        training_measurements_sha256=sha256(output / "training_measurements.npz"),
        native_measurements_sha256=sha256(output / "native_measurements.npz"),
        descriptors_sha256=sha256(output / "descriptors.npz"),
    )
    result["variants"] = {name: report["variants"][name] for name in ("clock", "timed")}
    result["variants"]["rgb"] = {
        "segments": diagnostics,
        "evaluation": evaluation,
        "report_sha256": sha256(output / "rgb/report.json"),
        "trajectory_sha256": sha256(output / "rgb/trajectory.npz"),
        "factor_archive_sha256": sha256(output / "rgb_factors.npz"),
    }
    for path, digest in protected.items():
        assert sha256(path) == digest
    assert code_hash() == frozen_code
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    aligned = [
        s["aligned_ate_translation_m"]
        for s in evaluation["segments"]
        if s["aligned_ate_translation_m"]["rmse"] is not None
    ]
    count = sum(s["count"] for s in aligned)
    ate_mm = (
        1000 * np.sqrt(sum(s["count"] * s["rmse"] ** 2 for s in aligned) / count)
        if count
        else None
    )
    print(
        "RESULT ATE mm",
        ate_mm,
        [
            (s["endpoint_translation_error_m"] * 1000, s["endpoint_rotation_error_deg"])
            for s in evaluation["segments"]
        ],
        flush=True,
    )


def run(parent, output):
    """Publish a fresh diagnostic while preserving every source packet."""
    parent, output = Path(parent), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rgb-geometry-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "result"
        _run(parent, stage)
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during publication")
        stage.rename(output)


def replay(output):
    """Rebuild association, factor and full-rate pose arrays without media."""
    output = Path(output)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    parent = Path(saved["parent_experiment"])
    if sha256(parent / "report.json") != saved["parent_report_sha256"]:
        raise ValueError("parent report changed")
    report, _ = verify_packets(parent)
    identity = saved["variants"]["rgb"]
    for path, digest in (
        (output / "training_measurements.npz", saved["training_measurements_sha256"]),
        (output / "native_measurements.npz", saved["native_measurements_sha256"]),
        (output / "descriptors.npz", saved["descriptors_sha256"]),
        (output / "rgb_factors.npz", identity["factor_archive_sha256"]),
        (output / "rgb/report.json", identity["report_sha256"]),
        (output / "rgb/trajectory.npz", identity["trajectory_sha256"]),
    ):
        if sha256(path) != digest:
            raise ValueError("saved RGB experiment input changed")
    cv2.setNumThreads(1)
    with np.load(parent / "training_measurements.npz", allow_pickle=False) as a:
        old = {k: a[k] for k in a.files}
    with np.load(output / "native_measurements.npz", allow_pickle=False) as a:
        native = {k: a[k] for k in a.files}
    with np.load(output / "descriptors.npz", allow_pickle=False) as a:
        descriptors = {int(k.removeprefix("frame_")): a[k] for k in a.files}
    graph, association = rgb_graph(old, descriptors)
    for key, value in association.items():
        if value != saved["rgb_association"][key]:
            raise AssertionError("RGB association diagnostic replay differs")
    with np.load(output / "training_measurements.npz", allow_pickle=False) as a:
        if set(a.files) != set(graph):
            raise AssertionError("RGB graph replay fields differ")
        for key, value in graph.items():
            np.testing.assert_array_equal(value, a[key])
    calibration = calibration_from_report(report)
    config = BatchConfig(**saved["config"])
    base, segments, packets = make_batch_packets(graph, native, calibration, config)
    poses, diagnostics, factors = fit_batch_packets(
        base, segments, packets, calibration.K_R, config
    )
    if diagnostics != identity["segments"]:
        raise AssertionError("RGB solver diagnostic replay differs")
    with np.load(output / "rgb_factors.npz", allow_pickle=False) as a:
        if set(a.files) != set(factors):
            raise AssertionError("RGB factor replay fields differ")
        for key, value in factors.items():
            np.testing.assert_array_equal(value, a[key])
    baseline, rows = load_trajectory(Path(report["baseline_run"]))
    motions, _ = motion_table(
        baseline,
        rows,
        report["split"]["train_indices"],
        calibration,
        TimingConfig(**report["timing_config"]),
    )
    repeated = export_color_poses(rows, poses, calibration, motions)
    _, original = load_trajectory(output / "rgb")
    np.testing.assert_array_equal(
        [r.T_SC_D for r in repeated], [r.T_SC_D for r in original]
    )
    return {
        "association_arrays_exact": True,
        "factor_arrays_exact": True,
        "diagnostics_exact": True,
        "full_rate_trajectory_exact": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    a = parser.parse_args()
    if a.replay:
        print(json.dumps(replay(a.output_dir), indent=2))
    elif a.fixed_run is None:
        parser.error("--fixed-run is required unless --replay is supplied")
    else:
        run(a.fixed_run, a.output_dir)
