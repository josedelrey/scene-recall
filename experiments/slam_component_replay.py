"""Offline component-wise BA with one preserved input gauge per component.

Disconnected cameras cannot acquire relative information from this solver.
Each observable component can refine its own relative geometry. Its first pose
stays fixed at the input value, without asserting cross-component certainty.
This experiment has no marginal prior, relocalization or map-merging mechanism.
"""

from dataclasses import replace

import numpy as np

from scene_recall.multiview.batch_ba import solve_batch
from scene_recall.multiview.joint_ba import landmark_layout


def solve_components(initial, factors, K, config=None):
    from slam_generalization import camera_components, subset

    ids, _, points = landmark_layout(factors)
    poses = initial.copy()
    records = []
    for component in camera_components(factors, len(initial)):
        mask = np.isin(factors.view, component)
        if len(component) < 2 or not mask.any():
            records.append(
                {
                    "cameras": component,
                    "status": "unchanged",
                    "reason": "single-camera gauge",
                }
            )
            continue
        packet = subset(factors, mask)
        packet = replace(
            packet, view=np.array([component.index(int(v)) for v in packet.view])
        )
        fitted, fitted_points, diagnostic = solve_batch(
            initial[component], packet, K, config
        )
        poses[component] = fitted
        points[np.searchsorted(ids, np.unique(packet.track_id))] = fitted_points
        records.append({"cameras": component, **diagnostic})
    return (
        poses,
        points,
        {
            "status": "refined"
            if any(r["status"] == "refined" for r in records)
            else "unchanged",
            "reason": "independent input gauges preserved",
            "gauge_roots": [r["cameras"][0] for r in records],
            "components": records,
        },
    )
