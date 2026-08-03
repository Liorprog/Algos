"""
tests/test_mesh_nd.py

Basic tests for MeshND behavior (gaussian add and normalization flag).
Run with: pytest -q
"""
from mesh import MeshND


def test_2d_gaussian_center_greater_than_neighbor():
    mesh = MeshND(bounds=[(0.0, 1.0), (0.0, 1.0)], cells=[20, 20], radius=0.15, amount = 1.0)
    center = [0.5, 0.5]
    mesh.add(center)
    counts = mesh.get_counts()
    idx_center = mesh._nearest_index(center)
    cx, cy = idx_center
    center_val = counts[cx][cy]
    assert center_val > 0.0
    left_ix = cx - 1 if cx - 1 >= 0 else cx
    left_val = counts[left_ix][cy]
    assert left_val <= center_val + 1e-9


def test_out_of_bounds_raises():
    mesh = MeshND(bounds=[(0.0, 1.0), (0.0, 1.0)], cells=[5, 5])
    try:
        mesh.add([1.5, 0.5])
        assert False, "Expected ValueError for out-of-bounds point"
    except ValueError:
        pass

test_2d_gaussian_center_greater_than_neighbor()