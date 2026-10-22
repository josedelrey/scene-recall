"""Reference-only scoring of frozen causal and revised component placements."""

import numpy as np

from scene_recall.odometry.evaluation import error_statistics, pose_error


def score_components(poses, components, references, indices=None):
    """Anchor each component at its first reference without aligning positions."""
    groups = {}
    components = np.asarray(components)
    for root in sorted(set(components)):
        selected = [
            i
            for i, component in enumerate(components)
            if component == root
            and references[i] is not None
            and (indices is None or i in indices)
        ]
        errors = []
        if selected:
            first = selected[0]
            alignment = poses[first] @ np.linalg.inv(references[first])
            errors = [pose_error(poses[i], alignment @ references[i]) for i in selected]
        groups[str(int(root))] = {
            "frames": int(np.sum(components == root)),
            "reference_frames": len(selected),
            "translation_m": error_statistics(error[0] for error in errors),
            "rotation_deg": error_statistics(error[1] for error in errors),
            "outside_10cm_5deg": sum(a > 0.1 or b > 5 for a, b in errors),
        }
    return {
        "component_count": len(groups),
        "root_frames": int(np.sum(components == components[0]))
        if len(components)
        else 0,
        "groups": groups,
    }


def latency_statistics(seconds):
    values = np.asarray(seconds, dtype=float)
    return {
        "count": len(values),
        "mean_s": float(values.mean()) if len(values) else None,
        "p50_s": float(np.percentile(values, 50)) if len(values) else None,
        "p95_s": float(np.percentile(values, 95)) if len(values) else None,
        "max_s": float(values.max()) if len(values) else None,
    }
