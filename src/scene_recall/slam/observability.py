"""Collective camera-motion information from fresh observation Schur blocks."""

import numpy as np

from scene_recall.multiview.pose_only import skew


def rigid_cut_information(blocks, views, inverse, inside, *, center=None):
    """Restrict the undamped Schur system to a group's common left motion.

    Landmarks are eliminated using all their observations. Summing a landmark's
    cross blocks before elimination preserves correlations between cameras in
    the group. A deficient cut proves a collective null direction in the
    ungauged system. Full rank does not prove global rigidity of either side.
    Internal deformation modes and sensor validity require separate checks.
    """
    B, _, D, _, E = blocks
    inside = np.asarray(inside, dtype=bool)
    if inside.shape != (len(B),):
        raise ValueError("cut must identify every camera")
    cross = np.zeros((len(D), 6, 3))
    mask = inside[views]
    np.add.at(cross, inverse[mask], E[mask])
    H = B[inside].sum(axis=0) - np.einsum(
        "nki,nij,nlj->kl", cross, np.linalg.inv(D), cross
    )
    H = (H + H.T) / 2
    # Center the rotational chart to reduce world-origin-dependent cancellation.
    chart = np.eye(6)
    if center is not None:
        chart[:3, 3:] = skew(np.asarray(center, dtype=float)[None])[0]
    H = chart.T @ H @ chart
    # Use the uneliminated energy for scaling. Scaling by the cut's own
    # diagonal would amplify cancellation noise in exactly null directions.
    energy = chart.T @ B[inside].sum(axis=0) @ chart
    energy_values, energy_vectors = np.linalg.eigh((energy + energy.T) / 2)
    energy_tolerance = max(float(energy_values[-1]) * 1e-12, 1e-15)
    retained = energy_values > energy_tolerance
    whitening = energy_vectors[:, retained] / np.sqrt(energy_values[retained])
    normalized = whitening.T @ H @ whitening
    # A centered one-point group has null columns even before elimination.
    # Do not normalize cancellation noise in those columns into information.
    values = np.sort(
        np.r_[
            np.zeros(6 - retained.sum()),
            np.linalg.eigvalsh((normalized + normalized.T) / 2),
        ]
    )
    tolerance = max(1e-8, float(values[-1]) * 1e-8)
    return {
        "camera_count": int(inside.sum()),
        "normalized_eigenvalues": values.tolist(),
        "rank": int(np.count_nonzero(values > tolerance)),
        "pose_energy_rank": int(retained.sum()),
        "pose_energy_tolerance": energy_tolerance,
        "tolerance": tolerance,
        "scope": "fresh undamped Schur restricted to collective rigid camera motion, landmarks eliminated jointly, no gauge or damping",
    }
