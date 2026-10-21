"""Reference-free direct descriptor checks of admitted incremental identities.

Agreement is conditional on both endpoints entering the graph. It is neither
ground-truth correspondence accuracy nor a false-match probability estimate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scene_recall.multiview.tracking import match_descriptors
from scene_recall.run_metadata import sha256


def audit(run, features):
    protocol = json.loads((run.parent / "protocol.json").read_text())
    if protocol["features_sha256"] != sha256(features):
        raise ValueError("features differ from the run's frozen measurements")
    with np.load(run / "measurements.npz", allow_pickle=False) as a:
        views = a.get("view", np.empty(0, dtype=np.int64)).copy()
        rows = a.get("feature_index", np.empty(0, dtype=np.int64)).copy()
        identities = a.get("frontend_identity", np.empty(0, dtype=np.int64)).copy()
    frame_ids = sorted(set(map(int, views)))
    owners = {
        key: dict(zip(rows[views == key], identities[views == key], strict=True))
        for key in frame_ids
    }
    counts, disagreements = {}, []
    with np.load(features, allow_pickle=False) as cache:
        for offset in (1, 2, 4):
            count = {
                "direct_matches": 0,
                "both_admitted": 0,
                "identity_disagreements": 0,
            }
            for i in range(offset, len(frame_ids)):
                source, target = frame_ids[i - offset], frame_ids[i]
                matches, _ = match_descriptors(
                    cache[f"{source}_descriptors"], cache[f"{target}_descriptors"]
                )
                count["direct_matches"] += len(matches)
                for query, train in matches:
                    if query in owners[source] and train in owners[target]:
                        count["both_admitted"] += 1
                        if owners[source][query] != owners[target][train]:
                            count["identity_disagreements"] += 1
                            disagreements.append(
                                [source, int(query), target, int(train)]
                            )
            counts[str(offset)] = count
    return {
        "measurements_sha256": sha256(run / "measurements.npz"),
        "features_sha256": sha256(features),
        "driver_sha256": sha256(Path(__file__)),
        "pair_offsets_in_admitted_keyframes": counts,
        "disagreements": disagreements,
        "interpretation": "Selected direct descriptor identity checks, without true correspondence labels or a false-match probability interpretation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    result = audit(args.run, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
