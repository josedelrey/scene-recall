"""Explicit experimental clock extraction from supported adapter metadata."""

import numpy as np


def exposure_times(dataset, provenance):
    """Read known adapter clock units without interpreting timestamps as exposures.

    Reported frame timestamps are assumed acquisition-time proxies. Neither
    exposure midpoints nor clock offsets are measured by these adapters.
    """
    if dataset == "tum_rgbd":
        scale = 1.0
    elif dataset == "scannet_v2":
        scale = 1e-6
    else:
        return None, None
    result = []
    for name in ("timestamp_color", "timestamp_depth"):
        value = provenance.get(name)
        result.append(
            float(value) * scale
            if value is not None
            and np.isfinite(value)
            and (dataset != "scannet_v2" or value > 0)
            else None
        )
    return tuple(result)
