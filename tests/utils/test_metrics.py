"""Comprehensive tests for metrics utilities."""

import numpy as np
import open3d as o3d
import pytest

from registration.utils.metrics import (
    compute_rmse_between_point_clouds,
    compute_rmse_transformations,
)


class TestComputeRMSEBetweenPointClouds:
    """Tests for compute_rmse_between_point_clouds function."""

    def test_identical_point_clouds(self, sample_point_cloud):
        """Test RMSE between identical point clouds is zero."""
        rmse, distances = compute_rmse_between_point_clouds(
            sample_point_cloud, sample_point_cloud
        )

        assert np.isclose(rmse, 0.0), "RMSE should be 0 for identical point clouds"
        assert np.allclose(distances, 0.0), "All distances should be 0"
        assert len(distances) == len(sample_point_cloud.points)

    def test_simple_translation(self, simple_point_cloud):
        """Test RMSE with simple translation."""
        source = simple_point_cloud
        target = o3d.geometry.PointCloud()

        # Translate all points by [1, 0, 0]
        target_points = np.asarray(source.points) + np.array([1.0, 0.0, 0.0])
        target.points = o3d.utility.Vector3dVector(target_points)

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        # All points moved by 1 unit, so RMSE should be 1.0
        assert np.isclose(rmse, 1.0), f"Expected RMSE=1.0, got {rmse}"
        assert np.allclose(distances, 1.0), "All distances should be 1.0"

    def test_known_rmse_value(self):
        """Test RMSE with known values."""
        # Create two point clouds with known RMSE
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Source: origin and [1, 0, 0]
        source.points = o3d.utility.Vector3dVector(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        )

        # Target: [0, 1, 0] and [1, 1, 0]
        # Distances: 1 and 1, so RMSE = sqrt(mean([1, 1])) = 1.0
        target.points = o3d.utility.Vector3dVector(
            np.array([[0, 1, 0], [1, 1, 0]], dtype=float)
        )

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        assert np.isclose(rmse, 1.0), f"Expected RMSE=1.0, got {rmse}"
        assert np.allclose(distances, [1.0, 1.0]), "Expected distances [1.0, 1.0]"

    @pytest.mark.parametrize(
        "offset,expected_rmse",
        [
            (np.array([0.0, 0.0, 0.0]), 0.0),  # No offset
            (np.array([1.0, 0.0, 0.0]), 1.0),  # X offset
            (np.array([0.0, 2.0, 0.0]), 2.0),  # Y offset
            (np.array([0.0, 0.0, 3.0]), 3.0),  # Z offset
            (np.array([3.0, 4.0, 0.0]), 5.0),  # Pythagorean triple (3-4-5)
        ],
    )
    def test_parametrized_offsets(self, simple_point_cloud, offset, expected_rmse):
        """Test RMSE with various offsets."""
        source = simple_point_cloud
        target = o3d.geometry.PointCloud()

        target_points = np.asarray(source.points) + offset
        target.points = o3d.utility.Vector3dVector(target_points)

        rmse, _ = compute_rmse_between_point_clouds(source, target)

        assert np.isclose(rmse, expected_rmse, atol=1e-6), (
            f"Expected RMSE={expected_rmse}, got {rmse}"
        )

    def test_different_number_of_points_raises_error(self):
        """Test that different point cloud sizes raise ValueError."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 1, 1]]))
        target.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))

        with pytest.raises(ValueError, match="same number of points"):
            compute_rmse_between_point_clouds(source, target)

    def test_empty_point_clouds(self):
        """Test RMSE with empty point clouds."""
        import warnings

        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Empty point clouds will trigger numpy warnings for mean of empty slice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rmse, distances = compute_rmse_between_point_clouds(source, target)

        # With empty arrays, mean of empty is NaN, sqrt(NaN) = NaN
        assert np.isnan(rmse) or rmse == 0.0, "RMSE of empty clouds should be NaN or 0"
        assert len(distances) == 0, "Distances array should be empty"

    def test_single_point(self):
        """Test RMSE with single-point clouds."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(np.array([[1, 2, 3]]))
        target.points = o3d.utility.Vector3dVector(np.array([[4, 6, 8]]))

        # Distance = sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9+16+25) = sqrt(50)
        expected_rmse = np.sqrt(50)

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        assert np.isclose(rmse, expected_rmse), f"Expected {expected_rmse}, got {rmse}"
        assert len(distances) == 1
        assert np.isclose(distances[0], expected_rmse)

    def test_large_point_cloud(self):
        """Test RMSE with larger point cloud."""
        # Create a cloud with 1000 random points
        np.random.seed(42)
        source_points = np.random.randn(1000, 3)
        target_points = source_points + 0.5  # Add constant offset

        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        target.points = o3d.utility.Vector3dVector(target_points)

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        # With offset [0.5, 0.5, 0.5], distance = sqrt(3*0.5^2) = sqrt(0.75)
        expected_rmse = np.sqrt(0.75)

        assert np.isclose(rmse, expected_rmse, atol=1e-6)
        assert len(distances) == 1000
        assert np.allclose(distances, expected_rmse, atol=1e-6)

    def test_mixed_distances(self):
        """Test RMSE with varying distances."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        # Create points with known distances: 0, 1, 2, 3
        source.points = o3d.utility.Vector3dVector(
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        )
        target.points = o3d.utility.Vector3dVector(
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        )

        # RMSE = sqrt(mean([0, 1, 4, 9])) = sqrt(14/4) = sqrt(3.5)
        expected_rmse = np.sqrt(3.5)
        expected_distances = np.array([0.0, 1.0, 2.0, 3.0])

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        assert np.isclose(rmse, expected_rmse, atol=1e-6)
        assert np.allclose(distances, expected_distances, atol=1e-6)

    def test_negative_coordinates(self):
        """Test RMSE with negative coordinates."""
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(np.array([[-1, -2, -3]]))
        target.points = o3d.utility.Vector3dVector(np.array([[1, 2, 3]]))

        # Distance = sqrt((1-(-1))^2 + (2-(-2))^2 + (3-(-3))^2)
        #          = sqrt(4 + 16 + 36) = sqrt(56)
        expected_rmse = np.sqrt(56)

        rmse, distances = compute_rmse_between_point_clouds(source, target)

        assert np.isclose(rmse, expected_rmse, atol=1e-6)


class TestComputeRMSETransformations:
    """Tests for compute_rmse_transformations function."""

    def test_identical_transformations(
        self, simple_point_cloud, identity_transformation
    ):
        """Test RMSE between identical transformations is zero."""
        T = identity_transformation

        rmse = compute_rmse_transformations(T, T, simple_point_cloud)

        assert np.isclose(rmse, 0.0), "RMSE should be 0 for identical transformations"

    def test_identity_vs_identity(self, simple_point_cloud):
        """Test that two identity transformations give zero RMSE."""
        T1 = np.eye(4)
        T2 = np.eye(4)

        rmse = compute_rmse_transformations(T1, T2, simple_point_cloud)

        assert np.isclose(rmse, 0.0)

    def test_translation_difference(self, simple_point_cloud):
        """Test RMSE with translation difference."""
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [1.0, 0.0, 0.0]  # Translate by 1 unit in X

        rmse = compute_rmse_transformations(T1, T2, simple_point_cloud)

        # All points should be 1 unit apart
        assert np.isclose(rmse, 1.0, atol=1e-6)

    def test_rotation_difference(self, simple_point_cloud):
        """Test RMSE with rotation difference."""
        T1 = np.eye(4)

        T2 = np.eye(4)
        # 90° rotation around Z-axis
        T2[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        rmse = compute_rmse_transformations(T1, T2, simple_point_cloud)

        # RMSE should be non-zero and depends on point distribution
        assert rmse > 0, "RMSE should be positive for different rotations"
        assert np.isfinite(rmse), "RMSE should be finite"

    @pytest.mark.parametrize(
        "translation,expected_rmse",
        [
            (np.array([0.0, 0.0, 0.0]), 0.0),
            (np.array([1.0, 0.0, 0.0]), 1.0),
            (np.array([0.0, 2.0, 0.0]), 2.0),
            (np.array([3.0, 4.0, 0.0]), 5.0),
        ],
    )
    def test_parametrized_translations(
        self, simple_point_cloud, translation, expected_rmse
    ):
        """Test RMSE with various translation offsets."""
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = translation

        rmse = compute_rmse_transformations(T1, T2, simple_point_cloud)

        assert np.isclose(rmse, expected_rmse, atol=1e-6)

    def test_combined_rotation_and_translation(self, simple_point_cloud):
        """Test RMSE with both rotation and translation."""
        T1 = np.eye(4)

        T2 = np.eye(4)
        # 90° rotation around Z + translation
        T2[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T2[:3, 3] = [1.0, 2.0, 3.0]

        rmse = compute_rmse_transformations(T1, T2, simple_point_cloud)

        assert rmse > 0, "RMSE should be positive"
        assert np.isfinite(rmse), "RMSE should be finite"

    def test_inverse_transformations(self, simple_point_cloud):
        """Test RMSE with inverse transformations."""
        # Create a transformation and its inverse
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T[:3, 3] = [1.0, 2.0, 3.0]

        T_inv = np.linalg.inv(T)

        # Applying T and then T_inv should give back original
        pcd_transformed = simple_point_cloud
        pcd_copy = o3d.geometry.PointCloud(pcd_transformed)
        pcd_copy.transform(T)
        pcd_copy.transform(T_inv)

        rmse, _ = compute_rmse_between_point_clouds(simple_point_cloud, pcd_copy)

        assert np.isclose(rmse, 0.0, atol=1e-6), "Inverse should undo transformation"

    def test_small_rotation_angle(self):
        """Test RMSE with small rotation angles."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))

        T1 = np.eye(4)
        T2 = np.eye(4)

        # Small rotation (1 degree around Z)
        angle = np.deg2rad(1)
        T2[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        rmse = compute_rmse_transformations(T1, T2, pcd)

        # For small angles, distance ≈ radius * angle
        # Point at [1, 0, 0] with 1° rotation: distance ≈ 1 * 0.0175 ≈ 0.0175
        expected_approx = 1 * angle  # Small angle approximation

        assert rmse > 0
        assert np.isclose(rmse, expected_approx, rtol=0.01)

    def test_known_rotation_rmse(self):
        """Test RMSE with 90° rotation for a single point."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))

        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° around Z

        rmse = compute_rmse_transformations(T1, T2, pcd)

        # Point [1,0,0] becomes [0,1,0], distance = sqrt(2)
        expected_rmse = np.sqrt(2)

        assert np.isclose(rmse, expected_rmse, atol=1e-6)

    def test_does_not_modify_original_point_cloud(self, simple_point_cloud):
        """Test that the function doesn't modify the input point cloud."""
        original_points = np.asarray(simple_point_cloud.points).copy()

        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [5.0, 5.0, 5.0]

        compute_rmse_transformations(T1, T2, simple_point_cloud)

        current_points = np.asarray(simple_point_cloud.points)

        assert np.allclose(original_points, current_points), (
            "Original point cloud should not be modified"
        )

    def test_scaling_transformation(self):
        """Test RMSE with scaling transformation."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0]]))

        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, :3] = np.eye(3) * 2  # Scale by 2

        rmse = compute_rmse_transformations(T1, T2, pcd)

        # Point [1,0,0] becomes [2,0,0], distance = 1
        # Point [0,1,0] becomes [0,2,0], distance = 1
        # RMSE = sqrt(mean([1, 1])) = 1.0
        assert np.isclose(rmse, 1.0, atol=1e-6)

    def test_complex_transformation_matrix(self):
        """Test with a complex transformation combining rotation, translation, and scale."""
        pcd = o3d.geometry.PointCloud()
        points = np.random.randn(10, 3)
        pcd.points = o3d.utility.Vector3dVector(points)

        # Complex transformation 1
        angle = np.deg2rad(45)
        R1 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = [1, 2, 3]

        # Complex transformation 2
        angle2 = np.deg2rad(30)
        R2 = np.array(
            [
                [np.cos(angle2), 0, np.sin(angle2)],
                [0, 1, 0],
                [-np.sin(angle2), 0, np.cos(angle2)],
            ]
        )
        T2 = np.eye(4)
        T2[:3, :3] = R2
        T2[:3, 3] = [2, 3, 4]

        rmse = compute_rmse_transformations(T1, T2, pcd)

        assert rmse > 0
        assert np.isfinite(rmse)

    def test_empty_point_cloud_transformations(self):
        """Test with empty point cloud."""
        import warnings

        pcd = o3d.geometry.PointCloud()
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [1, 2, 3]

        # Empty point cloud will trigger numpy warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rmse = compute_rmse_transformations(T1, T2, pcd)

        # Should handle empty cloud gracefully (NaN or 0)
        assert np.isnan(rmse) or rmse == 0.0
