"""Frozen, reference-free feasibility of covisibility-aware pending-ray admission."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from audit_shared_candidates import recover_allowance

from scene_recall.datasets.replay import open_source
from scene_recall.features import RGBDFeatures
from scene_recall.multiview.temporal import PairMotion, TimingConfig, bind_depth
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.observation_welding import ObservationWelding
from scene_recall.slam.online import OnlineConfig, OnlineSLAM
from scene_recall.slam.registration import register_anchor_pairs


def audit(run, output, frames, budget=128):
    if output.exists():
        raise ValueError("choose a fresh admission audit")
    if len(frames) < 2 or list(frames) != sorted(set(frames)) or budget < 1:
        raise ValueError(
            "require ordered source cameras, a later target and a positive budget"
        )
    report = json.loads((run / "report.json").read_text())
    saved = np.load(run / "shared_estimation.npz")
    reader, _ = open_source(report)
    config = report["online_config"]
    cv2.setNumThreads(1)
    engine = OnlineSLAM(
        reader.calibration,
        OnlineConfig(
            contrast=config["contrast"], feature_scale=config["feature_scale"]
        ),
    )
    calibration = reader.calibration
    selector = ObservationWelding(calibration)
    features, selected, native_indices = {}, {}, {}
    with reader:
        for observation in reader:
            frame = observation.frame_index
            if frame not in frames:
                continue
            extracted = engine.local.frontend._extract(observation, calibration)
            mask = saved["view"] == frame
            native = saved["native_index"][mask] >= 0
            if not native.any():
                raise ValueError(
                    f"frame {frame} lacks frozen native covariance support"
                )
            st, sr = recover_allowance(
                saved["native_points"][mask][native],
                saved["whitening_D"][mask][native],
                calibration.K_D,
            )
            E = saved["T_RcD"][mask][0]
            motion = PairMotion(
                E @ np.linalg.inv(calibration.T_RD),
                "recovered_owned_native",
                True,
                st,
                sr,
            )
            samples = bind_depth(
                extracted.pixels,
                observation.depth,
                calibration,
                motion,
                TimingConfig(**config["local"]["timing"]),
            )
            indices = saved["feature_index"][mask]
            np.testing.assert_array_equal(
                extracted.pixels[indices], saved["pixels"][mask]
            )
            np.testing.assert_array_equal(
                samples.source_index[indices], saved["native_index"][mask]
            )
            np.testing.assert_allclose(
                samples.whitening_D[indices][native],
                saved["whitening_D"][mask][native],
                atol=1e-8,
            )
            features[frame] = RGBDFeatures(
                extracted.pixels,
                extracted.descriptors,
                samples.points_R,
                extracted.descriptor_family,
            )
            native_indices[frame] = samples.source_index
            legacy = set(map(int, indices))
            extras = selector._spatial_order(
                extracted.pixels,
                [i for i in range(len(extracted.pixels)) if i not in legacy],
            )
            used = set(
                samples.source_index[indices][samples.source_index[indices] >= 0]
            )
            picks = []
            for i in extras:
                sample = int(samples.source_index[i])
                if sample >= 0 and sample in used:
                    continue
                picks.append(i)
                if sample >= 0:
                    used.add(sample)
                if len(picks) == budget:
                    break
            selected[frame] = legacy | set(picks)
    target = frames[-1]
    candidates = features[target]
    prioritized = set()
    for source in frames[:-1]:
        a = features[source]
        pairs, _ = match_descriptors(
            a.descriptors, candidates.descriptors, engine.registration.ratio
        )
        good = np.isfinite(a.points_R[pairs[:, 0]]).all(axis=1) & np.isin(
            pairs[:, 0], list(selected[source])
        )
        prioritized.update(map(int, pairs[good, 1]))
    legacy = set(map(int, saved["feature_index"][saved["view"] == target]))
    priority = selector._spatial_order(candidates.pixels, sorted(prioritized - legacy))
    remaining = selector._spatial_order(
        candidates.pixels,
        [i for i in range(len(candidates.pixels)) if i not in legacy | set(priority)],
    )
    used = set(native_indices[target][list(legacy)]) - {-1}
    cov = set(legacy)
    for i in priority + remaining:
        sample = int(native_indices[target][i])
        if sample >= 0 and sample in used:
            continue
        cov.add(i)
        if sample >= 0:
            used.add(sample)
        if len(cov) >= len(legacy) + budget:
            break
    rows = []
    for source in frames[:-1]:
        a = features[source]
        points = np.full_like(a.points_R, np.nan)
        indices = list(selected[source])
        points[indices] = a.points_R[indices]
        a = RGBDFeatures(a.pixels, a.descriptors, points, a.descriptor_family)
        pairs, _ = match_descriptors(
            a.descriptors, candidates.descriptors, engine.registration.ratio
        )
        for population, retained in [
            ("legacy", legacy),
            ("spatial", selected[target]),
            ("covisibility", cov),
            ("all_candidates", set(range(len(candidates.pixels)))),
        ]:
            owned = pairs[np.isin(pairs[:, 1], list(retained))]
            result = register_anchor_pairs(
                a, candidates, owned, calibration, engine.registration
            )
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "population": population,
                    "target_rows": len(retained),
                    "accepted": result.T_target_source is not None,
                    "verification": result.diagnostics,
                }
            )
    output.write_text(
        json.dumps(
            {
                "scope": "retrospective frozen motion and native uncertainty, exact retained binding verified, no reference poses, selection feasibility rather than causal accuracy",
                "package_sha256": code_hash(),
                "driver_sha256": sha256(Path(__file__)),
                "source_sha256": sha256(run / "shared_estimation.npz"),
                "frames": frames,
                "extra_budget": budget,
                "native_matched_target_candidates": len(prioritized),
                "registrations": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--budget", type=int, default=128)
    a = p.parse_args()
    audit(a.run, a.output, a.frames, a.budget)
