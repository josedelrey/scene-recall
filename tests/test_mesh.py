"""Independent plane/ray checks and a controlled disc-footprint counterexample."""

import numpy as np
import pytest
from helpers.mapping import pose

from scene_recall.geometry.camera import backproject_depth
from scene_recall.mapping.evaluation import render_depth
from scene_recall.mapping.mesh import TriangleMesh, render_mesh_depth, save_mesh
from scene_recall.mapping.tsdf import TSDFConfig, TSDFView, TSDFVolume


def test_mesh_inverse_pose_and_perspective_depth_match_analytic_plane():
    mesh = TriangleMesh(
        np.array([[-10, -10, 2], [10, -10, 2], [10, 10, 2], [-10, 10, 2]], dtype=float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    K = np.array([[20, 0, 20], [0, 20, 20], [0, 0, 1]], dtype=float)
    camera = pose(x=0.1, z=0.3, yaw=0.2)
    actual = render_mesh_depth(mesh, camera, K, (41, 41))
    rays = backproject_depth(np.ones((41, 41)), K)
    expected = (2 - camera[2, 3]) / (rays @ camera[:3, :3].T)[..., 2]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_depth_is_perspective_correct_and_nearest_without_face_culling():
    vertices = np.array([[0, 0, 1], [4, 0, 2], [0, 8, 4]], dtype=float)
    mesh = TriangleMesh(
        np.concatenate([vertices, vertices * 2]), np.array([[2, 1, 0], [3, 4, 5]])
    )
    result = render_mesh_depth(mesh, np.eye(4), np.eye(3), (3, 3))
    assert result[0, 1] == pytest.approx(4 / 3)
    assert result[1, 0] == pytest.approx(1.6)
    assert np.isnan(result[2, 2])


def test_near_clipping_matches_independent_ray_triangle_intersections():
    vertices = np.array([[-0.1, -0.2, -0.1], [0.5, -0.2, 1], [0.0, 0.7, 1]])
    mesh = TriangleMesh(vertices, np.array([[0, 1, 2]]))
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=float)
    actual = render_mesh_depth(mesh, np.eye(4), K, (11, 11), 0.05)
    expected = np.full((11, 11), np.nan)
    a, b, c = vertices
    e1, e2 = b - a, c - a
    for v in range(11):
        for u in range(11):
            ray = np.array([(u - 5) / 10, (v - 5) / 10, 1])
            cross = np.cross(ray, e2)
            denominator = np.dot(e1, cross)
            if abs(denominator) < 1e-12:
                continue
            p = np.dot(-a, cross) / denominator
            q = np.dot(ray, np.cross(-a, e1)) / denominator
            distance = np.dot(e2, np.cross(-a, e1)) / denominator
            if p >= -1e-10 and q >= -1e-10 and p + q <= 1 + 1e-10 and distance >= 0.05:
                expected[v, u] = distance
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_perfect_plane_disc_control_has_holes_while_connected_mesh_does_not():
    origin = np.array([-30, -30, 97])
    shape = (61, 61, 5)
    keys = np.indices(shape).reshape(3, -1).T + origin
    field = 2 - (keys[:, 2] + 0.5) * 0.02
    volume = TSDFVolume(
        TSDFConfig(), origin, shape, keys, field, np.ones(len(keys), dtype=np.uint16)
    )
    K = np.array([[200, 0, 50], [0, 200, 50], [0, 0, 1]], dtype=float)
    views = [TSDFView(0, np.eye(4), np.full((101, 101), 2.0))]
    surface, mesh = volume.surface(views, K), volume.mesh()
    discs = render_depth(surface, np.eye(4), K, (101, 101), 0.01)
    triangles = render_mesh_depth(mesh, np.eye(4), K, (101, 101))
    assert np.isfinite(discs).mean() < 0.85
    assert np.isfinite(triangles).all()
    np.testing.assert_allclose(discs[np.isfinite(discs)], 2, atol=1e-14)
    np.testing.assert_allclose(triangles, 2, atol=1e-14)


def test_empty_degenerate_and_invalid_meshes_and_portable_export(tmp_path):
    empty = TriangleMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert np.isnan(render_mesh_depth(empty, np.eye(4), np.eye(3), (3, 3))).all()
    degenerate = TriangleMesh(np.ones((3, 3)), np.array([[0, 1, 2]]))
    assert np.isnan(render_mesh_depth(degenerate, np.eye(4), np.eye(3), (3, 3))).all()
    with pytest.raises(ValueError, match="rigid"):
        render_mesh_depth(empty, np.zeros((4, 4)), np.eye(3), (3, 3))
    path = tmp_path / "mesh.ply"
    save_mesh(path, degenerate, np.full((3, 3), 120, dtype=np.uint8))
    payload = path.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"element vertex 3" in header and b"element face 1" in header
    assert len(body) == 3 * (3 * 8 + 3) + 1 + 3 * 4
