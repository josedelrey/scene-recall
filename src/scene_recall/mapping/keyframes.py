"""Optional exact-budget keyframes from estimated camera motion only."""

from itertools import pairwise

import numpy as np

from scene_recall.odometry.pipeline import is_rigid


def select_motion_keyframes(
    candidates: list[tuple[int, np.ndarray]],
    budget: int,
    rotation_radius_m: float = 1.0,
) -> tuple[list[int], dict]:
    """Spread distinct frames along cumulative translation plus angular travel.

    Candidates must already exclude held-out observations and lie in one
    connected chunk. Accept only IDs and estimated camera-to-segment poses,
    keeping references, depth and held-out poses outside this API. Endpoints
    are mandatory. A fixed metric lever arm converts radians to meters.
    Zero-motion chunks fall back to uniform candidate-index spacing. Constrain
    each nearest-quantile choice to leave room for the remaining distinct frames.
    Ties choose the earliest candidate. This is an offline path-length policy,
    sensitive to estimated pose noise and repeated traversal of the same scene.
    """
    n = len(candidates)
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or not 1 <= budget <= n
        or (n > 1 and budget < 2)
    ):
        raise ValueError("budget must fit the candidates and retain both endpoints")
    if not np.isfinite(rotation_radius_m) or rotation_radius_m <= 0:
        raise ValueError("rotation radius must be finite and positive")
    indices = [i for i, _ in candidates]
    if any(
        not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in indices
    ):
        raise ValueError("candidate IDs must be integers")
    if any(a >= b for a, b in pairwise(indices)):
        raise ValueError("candidate IDs must be strictly increasing")
    poses = np.array([pose for _, pose in candidates])
    if any(not is_rigid(pose) for pose in poses):
        raise ValueError("candidate poses must be rigid")
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    relative = poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
    travel = np.r_[0.0, np.cumsum(translation + rotation_radius_m * rotation)]
    zero_motion = bool(travel[-1] <= 1e-12)
    coordinate = np.arange(n, dtype=float) if zero_motion else travel
    targets = np.linspace(0, coordinate[-1], budget)
    selected = [0]
    for j in range(1, budget - 1):
        low, stop = selected[-1] + 1, n - (budget - j) + 1
        selected.append(low + int(np.argmin(np.abs(coordinate[low:stop] - targets[j]))))
    if budget > 1:
        selected.append(n - 1)
    return [indices[i] for i in selected], {
        "candidate_indices": indices,
        "candidate_cumulative_travel_m": travel.tolist(),
        "selected_candidate_positions": selected,
        "budget": budget,
        "rotation_radius_m": rotation_radius_m,
        "translation_travel_m": float(translation.sum()),
        "rotation_travel_rad": float(rotation.sum()),
        "combined_travel_m": float(travel[-1]),
        "zero_motion_fallback": zero_motion,
    }
