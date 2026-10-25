"""Two-level additive Schur preconditioning with a bounded coarse space."""

from itertools import pairwise

import numpy as np


class CoarsePreconditioner:
    """Block-Jacobi plus exact Galerkin correction of aggregate pose modes.

    Constant left-pose increments on adjacent free-camera aggregates form Z.
    The coarse matrix is Z.T S Z, assembled from the same landmark blocks.
    Both terms are positive definite for an SPD damped Schur system. This
    changes the iteration path only, never the linear system being solved.
    """

    def __init__(self, operator, group_size=8, max_groups=48):
        self.diagonal = np.linalg.inv(operator.diagonal)
        count = len(operator.free)
        size = max(group_size, int(np.ceil(count / max_groups)))
        self.groups = np.arange(count) // size
        groups = int(self.groups[-1]) + 1 if count else 0
        S = np.zeros((groups * 6, groups * 6))
        for camera, group in zip(operator.free, self.groups, strict=True):
            block = slice(6 * group, 6 * group + 6)
            S[block, block] += operator.B[camera]
        # A landmark can touch an aggregate through multiple observations.
        # Sum its cross blocks before elimination, retaining those correlations.
        labels = np.full(len(operator.B), -1, dtype=int)
        labels[operator.free] = self.groups
        mask = labels[operator.views] >= 0
        pairs = np.column_stack((operator.inverse[mask], labels[operator.views[mask]]))
        if len(pairs):
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            cross = np.zeros((len(unique), 6, 3))
            np.add.at(cross, inverse, operator.cross[mask])
            starts = np.r_[0, np.flatnonzero(np.diff(unique[:, 0])) + 1, len(unique)]
            for start, stop in pairwise(starts):
                point = unique[start, 0]
                columns = (6 * unique[start:stop, 1, None] + np.arange(6)).ravel()
                X = cross[start:stop].reshape(-1, 3)
                S[np.ix_(columns, columns)] -= X @ operator.Ci[point] @ X.T
        S = (S + S.T) / 2
        # Cholesky certifies the coarse SPD assumption before constructing its
        # inverse once. PCG applications use matrix products, not refactorization.
        L = np.linalg.cholesky(S)
        Li = np.linalg.solve(L, np.eye(len(L)))
        self.coarse_inverse = Li.T @ Li
        self.matrix = S

    def __call__(self, x):
        rows = x.reshape(-1, 6)
        result = np.einsum("nij,nj->ni", self.diagonal, rows)
        restricted = np.zeros((len(self.coarse_inverse) // 6, 6))
        np.add.at(restricted, self.groups, rows)
        correction = (self.coarse_inverse @ restricted.ravel()).reshape(-1, 6)
        result += correction[self.groups]
        return result.ravel()

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for value in (self.diagonal, self.groups, self.coarse_inverse, self.matrix)
        )
