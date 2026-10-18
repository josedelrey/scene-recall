"""Rigid-pose checks preserve inputs and respect each consumer's tolerance."""

import numpy as np
import pytest

from scene_recall.geometry.transforms import is_rigid


def test_rigid_transform_with_rotation_and_translation_is_preserved():
    transform = np.array(
        [[0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=float
    )
    original = transform.copy()
    assert is_rigid(transform, atol=1e-6)
    np.testing.assert_array_equal(transform, original)


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([-1.0, 1, 1, 1]),
        np.diag([2.0, 1, 1, 1]),
        np.diag([1.0, 1, 1, 1.000001]),
    ],
)
def test_nonrigid_or_malformed_transforms_are_rejected_without_repair(transform):
    original = transform.copy()
    assert not is_rigid(transform)
    np.testing.assert_array_equal(transform, original)


def test_consumer_tolerance_is_explicit_without_modifying_rotation():
    transform = np.eye(4)
    transform[0, 0] += 1e-5
    original = transform.copy()
    assert is_rigid(transform)
    assert not is_rigid(transform, atol=1e-6)
    np.testing.assert_array_equal(transform, original)
