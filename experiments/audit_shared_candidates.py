"""Frozen native evidence omitted before shared-ledger admission.

Recover the frozen motion allowance from owned native whitening, reproduce the
original binding on every retained feature, then inspect additional raw image
features and disjoint native registration. No reference poses enter this audit.
This is retrospective feasibility evidence, not causal SLAM accuracy.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from scene_recall.datasets.tum import TUMConfig, TUMReader
from scene_recall.features import RGBDFeatures
from scene_recall.multiview.temporal import (
    PairMotion,
    TimingConfig,
    bind_depth,
    native_whitening,
)
from scene_recall.multiview.tracking import match_descriptors
from scene_recall.run_metadata import code_hash, sha256
from scene_recall.slam.online import OnlineConfig, OnlineSLAM
from scene_recall.slam.registration import register_anchor, register_correspondences


def recover_allowance(points, whitening, K):
    baseline = native_whitening(points, K, PairMotion(np.eye(4), "zero", True))
    base, measured = np.linalg.inv(baseline), np.linalg.inv(whitening)
    covariance = measured @ measured.transpose(0, 2, 1) - base @ base.transpose(0, 2, 1)
    unit = points / np.linalg.norm(points, axis=1)[:, None]
    translation = np.einsum("ni,nij,nj->n", unit, covariance, unit)
    rotation = (np.trace(covariance, axis1=1, axis2=2) - 3 * translation) / (
        2 * np.sum(points**2, axis=1)
    )
    return float(np.sqrt(max(0, np.median(translation)))), float(
        np.sqrt(max(0, np.median(rotation)))
    )


def audit(run, output, frames):
    if output.exists():
        raise ValueError("choose a fresh candidate audit directory")
    report = json.loads((run / "report.json").read_text())
    if report["source"]["dataset"] != "tum_rgbd":
        raise ValueError("candidate audit currently supports TUM")
    data = np.load(run / "shared_estimation.npz")
    frames = sorted(set(frames))
    if any(frame not in data["view"] for frame in frames):
        raise ValueError("audit frames require recorded native packets")
    output.mkdir(parents=True)
    config = report["online_config"]
    source = report["source"]
    reader = TUMReader(
        source["path"],
        start=source["start"],
        stop=source["start"] + max(frames) + 1,
        config=TUMConfig(**source["association_config"]),
    )
    frontend = OnlineSLAM(
        reader.calibration,
        OnlineConfig(
            contrast=config["contrast"], feature_scale=config["feature_scale"]
        ),
    ).local.frontend
    retained, candidates, rows = {}, {}, []
    cv2.setNumThreads(1)
    with reader:
        for observation in reader:
            frame = observation.frame_index
            if frame not in frames:
                continue
            features = frontend._extract(observation, reader.calibration)
            mask = data["view"] == frame
            indices, native = data["feature_index"][mask], data["native_index"][mask]
            np.testing.assert_array_equal(
                features.pixels[indices], data["pixels"][mask]
            )
            valid = native >= 0
            translation, rotation = recover_allowance(
                data["native_points"][mask][valid],
                data["whitening_D"][mask][valid],
                reader.calibration.K_D,
            )
            exposure = data["T_RcD"][mask][0]
            motion = PairMotion(
                exposure @ np.linalg.inv(reader.calibration.T_RD),
                "recovered_from_owned_native_whitening",
                True,
                translation,
                rotation,
            )
            samples = bind_depth(
                features.pixels,
                observation.depth,
                reader.calibration,
                motion,
                TimingConfig(**config["local"]["timing"]),
            )
            np.testing.assert_array_equal(samples.source_index[indices], native)
            np.testing.assert_allclose(
                samples.whitening_D[indices][valid],
                data["whitening_D"][mask][valid],
                atol=1e-8,
            )
            candidate = RGBDFeatures(
                features.pixels,
                features.descriptors,
                samples.points_R,
                features.descriptor_family,
            )
            restricted = np.full_like(samples.points_R, np.nan)
            restricted[indices[valid]] = samples.points_R[indices[valid]]
            retained[frame] = RGBDFeatures(
                features.pixels,
                features.descriptors,
                restricted,
                features.descriptor_family,
            )
            candidates[frame] = candidate
            rows.append(
                {
                    "frame": frame,
                    "retained_rows": int(mask.sum()),
                    "retained_native": int(valid.sum()),
                    "all_features": len(features.pixels),
                    "all_bound_native": int((samples.source_index >= 0).sum()),
                    "recovered_translation_sigma_m": translation,
                    "recovered_rotation_sigma_rad": rotation,
                }
            )
    registrations, one_sided = [], []
    for source, target in pairwise(frames):
        for population, collection in [
            ("retained", retained),
            ("all_bound_candidates", candidates),
        ]:
            a, b = collection[source], collection[target]
            pairs, _ = match_descriptors(
                a.descriptors, b.descriptors, frontend.registration.ratio
            )
            pairs = pairs[
                np.isfinite(a.points_R[pairs[:, 0]]).all(axis=1)
                & np.isfinite(b.points_R[pairs[:, 1]]).all(axis=1)
            ]
            i, j = pairs.T
            result = register_correspondences(
                a.points_R[i],
                b.points_R[j],
                a.pixels[i],
                b.pixels[j],
                reader.calibration.K_R,
                reader.calibration.rgb_shape,
                frontend.registration,
            )
            registrations.append(
                {
                    "source": source,
                    "target": target,
                    "population": population,
                    "metric_pairs": len(pairs),
                    "accepted": result.T_target_source is not None,
                    "verification": result.diagnostics,
                }
            )
        for anchor, bearing in ((source, target), (target, source)):
            result = register_anchor(
                candidates[anchor],
                candidates[bearing],
                reader.calibration,
                frontend.registration,
            )
            one_sided.append(
                {
                    "native_anchor": anchor,
                    "bearing_target": bearing,
                    "accepted": result.T_target_source is not None,
                    "verification": result.diagnostics,
                }
            )
    (output / "report.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(run / "shared_estimation.npz"),
                "package_sha256": code_hash(),
                "driver_sha256": sha256(Path(__file__)),
                "scope": "frozen exposure and recovered native uncertainty, exact binding verified on retained rows, no reference poses or causal feedback, extra candidates are not added to an estimator",
                "frames": rows,
                "registrations": registrations,
                "one_sided_registrations": one_sided,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    audit(args.run, args.output, args.frames)


if __name__ == "__main__":
    main()
