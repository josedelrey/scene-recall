"""Block sparse normal equations with a checked preconditioned CG solve."""

import numpy as np


class BlockSystem:
    """Store only observed 6-by-6 blocks, without a dense graph Hessian."""

    def __init__(self, size):
        self.size = size
        self.blocks = {}

    def add(self, a, b, value):
        key = a.start // 6, b.start // 6
        if key not in self.blocks:
            self.blocks[key] = np.zeros((6, 6))
        self.blocks[key] += value

    def solve(self, rhs, damping, *, tolerance=1e-8, max_iterations=1000):
        n = self.size // 6
        diagonal = np.array([self.blocks[i, i] for i in range(n)])
        regularizer = damping * np.maximum(
            np.diagonal(diagonal, axis1=1, axis2=2), 1e-8
        )
        diagonal = diagonal.copy()
        diagonal[:, np.arange(6), np.arange(6)] += regularizer
        inverse = np.linalg.inv(diagonal)
        keys = list(self.blocks)
        rows = np.array([k[0] for k in keys])
        columns = np.array([k[1] for k in keys])
        blocks = np.array([self.blocks[k] for k in keys])

        def multiply(x):
            result = regularizer * x
            np.add.at(result, rows, np.einsum("nij,nj->ni", blocks, x[columns]))
            return result

        def precondition(x):
            return np.einsum("nij,nj->ni", inverse, x)

        b = rhs.reshape(n, 6)
        x = np.zeros_like(b)
        r = b.copy()
        threshold = tolerance * max(float(np.linalg.norm(b)), 1e-12)
        z = precondition(r)
        p, rz = z.copy(), float(np.sum(r * z))
        for _ in range(max_iterations):
            # Recursive residuals can drift, so check the actual system.
            if (
                np.linalg.norm(r) <= threshold
                and np.linalg.norm(b - multiply(x)) <= threshold * 2
            ):
                return x.ravel()
            Ap = multiply(p)
            denominator = float(np.sum(p * Ap))
            if not np.isfinite(denominator) or denominator <= 0 or rz <= 0:
                break
            alpha = rz / denominator
            x += alpha * p
            r -= alpha * Ap
            z = precondition(r)
            updated = float(np.sum(r * z))
            p = z + (updated / rz) * p
            rz = updated
        raise np.linalg.LinAlgError("block PCG did not reach its residual tolerance")
