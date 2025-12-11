"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import open3d as o3d


@pytest.fixture
def sample_point_cloud():
    """Create a simple sample point cloud for testing."""
    # Create a cube point cloud
    points = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                points.append([x, y, z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd


@pytest.fixture
def simple_point_cloud():
    """Create a very simple point cloud with 4 points for easier testing."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def identity_transformation():
    """Return a 4x4 identity transformation matrix."""
    return np.eye(4)


@pytest.fixture
def sample_rotation_matrix():
    """Return a sample 3x3 rotation matrix (90° around z-axis)."""
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


@pytest.fixture
def sample_transformation_matrix():
    """Return a sample 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T[:3, 3] = [1.0, 2.0, 3.0]
    return T


def create_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    Helper function to create rotation matrices.

    Args:
        axis: Rotation axis ('x', 'y', or 'z')
        angle_deg: Rotation angle in degrees

    Returns:
        3x3 rotation matrix
    """
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)

    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
