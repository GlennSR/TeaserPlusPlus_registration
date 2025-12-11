"""Unit tests for point cloud geometry utilities."""

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from registration.utils.point_cloud import (
    rough_scale_point_cloud,
    rough_scale_point_cloud_from_file,
    align_centers,
    align_centers_from_files,
)


# Helper function to create test point clouds
def create_test_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from numpy array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


class TestRoughScalePointCloud:
    """Tests for rough_scale_point_cloud function."""

    def test_unit_cube(self):
        """Test scale estimation for a unit cube."""
        # Create points forming a unit cube (0 to 1)
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Unit cube has max extent ~1.7 (diagonal), floor(log10(1.7)) = 0
        assert scale == 1.0, f"Expected scale 1.0 for unit cube, got {scale}"

    def test_large_point_cloud(self):
        """Test scale estimation for a large point cloud."""
        # Create points spanning 0 to 100 in each dimension
        points = np.array(
            [
                [0, 0, 0],
                [100, 0, 0],
                [0, 100, 0],
                [0, 0, 100],
                [100, 100, 100],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Max extent ~173 (diagonal), floor(log10(173)) = 2
        assert scale == 100.0, f"Expected scale 100.0 for large cloud, got {scale}"

    def test_small_point_cloud(self):
        """Test scale estimation for a small point cloud."""
        # Create points spanning 0 to 0.01
        points = np.array(
            [
                [0, 0, 0],
                [0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.01],
                [0.01, 0.01, 0.01],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Max extent ~0.017, floor(log10(0.017)) = -2
        assert scale == 0.01, f"Expected scale 0.01 for small cloud, got {scale}"

    def test_single_point(self):
        """Test scale estimation for a single point."""
        points = np.array([[1.0, 2.0, 3.0]])
        pcd = create_test_point_cloud(points)

        # Open3D requires at least 4 points for oriented bounding box
        with pytest.raises(RuntimeError, match="not enough points"):
            _ = rough_scale_point_cloud(pcd)

    def test_collinear_points(self):
        """Test scale estimation for points that would be degenerate."""
        # Create a proper 3D tetrahedron (4 non-coplanar points)
        points = np.array(
            [
                [0, 0, 0],
                [15, 0, 0],
                [0, 15, 0],
                [0, 0, 15],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Max extent is ~15-21, floor(log10(15-21)) = 1
        assert scale == 10.0, f"Expected scale 10.0, got {scale}"

    def test_negative_coordinates(self):
        """Test scale estimation with negative coordinates."""
        # Create proper 3D tetrahedron with negative coordinates (non-coplanar)
        points = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            dtype=np.float64,
        )
        # Offset to negative coordinates
        points -= 5
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Max extent is ~15, floor(log10(15)) = 1
        assert scale == 10.0, f"Expected scale 10.0, got {scale}"

    def test_random_cloud(self):
        """Test scale estimation with random point cloud."""
        np.random.seed(42)
        points = np.random.randn(100, 3) * 25  # Random cloud with std=25
        pcd = create_test_point_cloud(points)

        scale = rough_scale_point_cloud(pcd)

        # Should return a power of 10
        assert scale in [0.1, 1.0, 10.0, 100.0, 1000.0], (
            f"Scale should be a power of 10, got {scale}"
        )


class TestRoughScalePointCloudFromFile:
    """Tests for rough_scale_point_cloud_from_file function."""

    def test_load_and_scale_ply_file(self):
        """Test loading PLY file and computing scale."""
        # Create a temporary PLY file
        points = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [0, 10, 0],
                [0, 0, 10],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
            o3d.io.write_point_cloud(tmp_path, pcd)

        try:
            scale = rough_scale_point_cloud_from_file(tmp_path)
            assert scale == 10.0, f"Expected scale 10.0, got {scale}"
        finally:
            Path(tmp_path).unlink()

    def test_load_and_scale_pcd_file(self):
        """Test loading PCD file and computing scale."""
        points = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
        pcd = create_test_point_cloud(points)

        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as tmp:
            tmp_path = tmp.name
            o3d.io.write_point_cloud(tmp_path, pcd)

        try:
            scale = rough_scale_point_cloud_from_file(tmp_path)
            assert scale == 1.0, f"Expected scale 1.0, got {scale}"
        finally:
            Path(tmp_path).unlink()

    def test_consistency_with_direct_function(self):
        """Test that loading from file gives same result as direct call."""
        points = np.random.randn(50, 3) * 15
        pcd = create_test_point_cloud(points)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
            o3d.io.write_point_cloud(tmp_path, pcd)

        try:
            scale_direct = rough_scale_point_cloud(pcd)
            scale_from_file = rough_scale_point_cloud_from_file(tmp_path)

            assert scale_direct == scale_from_file, (
                "Scale from file should match direct computation"
            )
        finally:
            Path(tmp_path).unlink()


class TestAlignCenters:
    """Tests for align_centers function."""

    def test_identical_centered_clouds(self):
        """Test alignment of two identical centered point clouds."""
        points = np.array(
            [
                [-1, -1, -1],
                [1, 1, 1],
                [-1, 1, -1],
                [1, -1, 1],
            ],
            dtype=np.float64,
        )

        source = create_test_point_cloud(points)
        target = create_test_point_cloud(points)

        T = align_centers(source, target)

        # Should return near-identity translation (zero translation)
        assert T.shape == (4, 4), "Output should be 4x4 matrix"
        assert np.allclose(T[:3, :3], np.eye(3)), "Rotation part should be identity"
        assert np.allclose(T[:3, 3], 0, atol=1e-10), "Translation should be zero"
        assert np.allclose(T[3, :], [0, 0, 0, 1]), "Bottom row should be [0, 0, 0, 1]"

    def test_simple_translation_alignment(self):
        """Test alignment of offset point clouds."""
        source_points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Target is translated by [5, 3, 2]
        target_points = source_points + np.array([5, 3, 2])

        source = create_test_point_cloud(source_points)
        target = create_test_point_cloud(target_points)

        T = align_centers(source, target)

        # Translation should approximately be [5, 3, 2]
        assert np.allclose(T[:3, 3], [5, 3, 2], atol=1e-10), (
            f"Expected translation [5, 3, 2], got {T[:3, 3]}"
        )

    def test_transformation_matrix_structure(self):
        """Test that output has correct transformation matrix structure."""
        source = create_test_point_cloud(np.random.randn(10, 3))
        target = create_test_point_cloud(np.random.randn(10, 3))

        T = align_centers(source, target)

        assert T.shape == (4, 4), "Should be 4x4 matrix"
        assert np.allclose(T[:3, :3], np.eye(3)), "Should be pure translation"
        assert np.allclose(T[3, :3], 0), "Bottom row first 3 elements should be zero"
        assert np.isclose(T[3, 3], 1.0), "Bottom-right should be 1"

    def test_with_initial_transformation(self):
        """Test alignment with initial transformation applied to source."""
        source_points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )

        target_points = np.array(
            [
                [5, 5, 5],
                [6, 5, 5],
                [5, 6, 5],
            ],
            dtype=np.float64,
        )

        source = create_test_point_cloud(source_points)
        target = create_test_point_cloud(target_points)

        # Apply initial translation to source
        trans_init = np.eye(4)
        trans_init[:3, 3] = [2, 2, 2]

        T = align_centers(source, target, trans_init=trans_init)

        # Should compute alignment after initial transformation
        assert T.shape == (4, 4), "Should be 4x4 matrix"
        assert np.allclose(T[:3, :3], np.eye(3)), "Should be pure translation"

    def test_with_correction_transformation(self):
        """Test alignment with correction transformation."""
        source_points = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
            ],
            dtype=np.float64,
        )

        target_points = np.array(
            [
                [0, 1, 0],
                [0, 2, 0],
            ],
            dtype=np.float64,
        )

        source = create_test_point_cloud(source_points)
        target = create_test_point_cloud(target_points)

        # Correction that rotates 90 degrees around z-axis
        correction = np.eye(4)
        correction[:3, :3] = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )

        T = align_centers(source, target, correction=correction)

        assert T.shape == (4, 4), "Should be 4x4 matrix"

    def test_modifies_input_clouds_warning(self):
        """Test that function modifies input point clouds (as documented)."""
        source_points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        target_points = np.array([[5, 0, 0], [6, 0, 0]], dtype=np.float64)

        source = create_test_point_cloud(source_points)
        target = create_test_point_cloud(target_points)

        # Store original points
        original_source = np.asarray(source.points).copy()
        original_target = np.asarray(target.points).copy()

        # Apply transformation with non-identity correction
        correction = np.eye(4)
        correction[:3, 3] = [1, 1, 1]

        _ = align_centers(source, target, correction=correction)

        # Points should be modified
        modified_source = np.asarray(source.points)
        modified_target = np.asarray(target.points)

        assert not np.allclose(modified_source, original_source), (
            "Source should be modified in-place"
        )
        assert not np.allclose(modified_target, original_target), (
            "Target should be modified in-place"
        )

    def test_negative_coordinates(self):
        """Test alignment with negative coordinates."""
        source = create_test_point_cloud(np.array([[-5, -5, -5], [-3, -3, -3]]))
        target = create_test_point_cloud(np.array([[5, 5, 5], [7, 7, 7]]))

        T = align_centers(source, target)

        assert T.shape == (4, 4), "Should be 4x4 matrix"
        # Translation should move from negative to positive region
        assert T[:3, 3].sum() > 0, "Should translate to positive direction"

    def test_large_point_clouds(self):
        """Test alignment with large point clouds."""
        np.random.seed(42)
        source = create_test_point_cloud(np.random.randn(1000, 3))
        target = create_test_point_cloud(np.random.randn(1000, 3) + 10)

        T = align_centers(source, target)

        assert T.shape == (4, 4), "Should be 4x4 matrix"
        assert np.allclose(T[:3, :3], np.eye(3)), "Should be pure translation"


class TestAlignCentersFromFiles:
    """Tests for align_centers_from_files function."""

    def test_align_from_files(self):
        """Test alignment of point clouds loaded from files."""
        source_points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )

        target_points = np.array(
            [
                [10, 10, 10],
                [11, 10, 10],
                [10, 11, 10],
            ],
            dtype=np.float64,
        )

        source_pcd = create_test_point_cloud(source_points)
        target_pcd = create_test_point_cloud(target_points)

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as src_tmp:
            source_file = src_tmp.name
            o3d.io.write_point_cloud(source_file, source_pcd)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tgt_tmp:
            target_file = tgt_tmp.name
            o3d.io.write_point_cloud(target_file, target_pcd)

        try:
            T = align_centers_from_files(source_file, target_file)

            assert T.shape == (4, 4), "Should be 4x4 matrix"
            assert np.allclose(T[:3, :3], np.eye(3)), "Should be pure translation"
            # Translation should be approximately [10, 10, 10]
            assert np.allclose(T[:3, 3], [10, 10, 10], atol=0.1), (
                f"Expected translation ~[10, 10, 10], got {T[:3, 3]}"
            )
        finally:
            Path(source_file).unlink()
            Path(target_file).unlink()

    def test_consistency_with_direct_function(self):
        """Test that file loading gives same result as direct call."""
        np.random.seed(123)
        source_points = np.random.randn(20, 3)
        target_points = np.random.randn(20, 3) + 5

        source_pcd = create_test_point_cloud(source_points)
        target_pcd = create_test_point_cloud(target_points)

        # Save to files
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as src_tmp:
            source_file = src_tmp.name
            o3d.io.write_point_cloud(source_file, source_pcd)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tgt_tmp:
            target_file = tgt_tmp.name
            o3d.io.write_point_cloud(target_file, target_pcd)

        try:
            # Get transformation from files
            T_from_files = align_centers_from_files(source_file, target_file)

            # Get transformation from direct call
            source_direct = create_test_point_cloud(source_points)
            target_direct = create_test_point_cloud(target_points)
            T_direct = align_centers(source_direct, target_direct)

            assert np.allclose(T_from_files, T_direct, atol=1e-10), (
                "Results should match between file and direct methods"
            )
        finally:
            Path(source_file).unlink()
            Path(target_file).unlink()

    def test_with_transformations_from_files(self):
        """Test file-based alignment with initial and correction transforms."""
        source_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        target_points = np.array([[5, 5, 5], [6, 6, 6]], dtype=np.float64)

        source_pcd = create_test_point_cloud(source_points)
        target_pcd = create_test_point_cloud(target_points)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as src_tmp:
            source_file = src_tmp.name
            o3d.io.write_point_cloud(source_file, source_pcd)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tgt_tmp:
            target_file = tgt_tmp.name
            o3d.io.write_point_cloud(target_file, target_pcd)

        try:
            trans_init = np.eye(4)
            trans_init[:3, 3] = [1, 0, 0]

            correction = np.eye(4)
            correction[:3, 3] = [0, 1, 0]

            T = align_centers_from_files(
                source_file, target_file, trans_init=trans_init, correction=correction
            )

            assert T.shape == (4, 4), "Should be 4x4 matrix"
            assert np.allclose(T[:3, :3], np.eye(3)), "Should be pure translation"
        finally:
            Path(source_file).unlink()
            Path(target_file).unlink()


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_scale_and_align_workflow(self):
        """Test typical workflow: compute scale, then align centers."""
        # Create source and target with different scales and positions
        source_points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        target_points = np.array(
            [
                [100, 100, 100],
                [110, 100, 100],
                [100, 110, 100],
                [100, 100, 110],
            ],
            dtype=np.float64,
        )

        source = create_test_point_cloud(source_points)
        target = create_test_point_cloud(target_points)

        # Compute scales
        scale_source = rough_scale_point_cloud(source)
        scale_target = rough_scale_point_cloud(target)

        assert scale_source == 1.0, "Source scale should be 1.0"
        assert scale_target == 10.0, "Target scale should be 10.0"

        # Align centers
        T = align_centers(source, target)

        assert T.shape == (4, 4), "Alignment should produce 4x4 matrix"

    def test_full_file_based_workflow(self):
        """Test complete workflow using file operations."""
        points = np.random.randn(50, 3) * 10
        pcd = create_test_point_cloud(points)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
            o3d.io.write_point_cloud(tmp_path, pcd)

        try:
            # Compute scale from file
            scale = rough_scale_point_cloud_from_file(tmp_path)
            assert isinstance(scale, float), "Scale should be float"
            assert scale > 0, "Scale should be positive"

            # Create second file for alignment test
            points2 = np.random.randn(50, 3) * 10 + 20
            pcd2 = create_test_point_cloud(points2)

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp2:
                tmp_path2 = tmp2.name
                o3d.io.write_point_cloud(tmp_path2, pcd2)

            try:
                # Align from files
                T = align_centers_from_files(tmp_path, tmp_path2)
                assert T.shape == (4, 4), "Should produce 4x4 transformation"
            finally:
                Path(tmp_path2).unlink()
        finally:
            Path(tmp_path).unlink()
