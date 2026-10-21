"""Local descriptor voting for a bounded place bank, without learned vocabulary.

Votes prioritize usable source-depth correspondences. They propose candidates
only. A high score is never a geometric acceptance or a place identity proof.
"""

import numpy as np

from scene_recall.multiview.tracking import match_descriptors


def descriptor_votes(source, target, ratio=0.75):
    pairs, _ = match_descriptors(source.descriptors, target.descriptors, ratio)
    metric = np.isfinite(source.points_R[pairs[:, 0]]).all(axis=1)
    return int(metric.sum()), len(pairs)
