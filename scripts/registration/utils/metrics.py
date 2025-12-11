"""Metrics for point cloud registration evaluation."""

import copy
import logging

import numpy as np
import open3d as o3d


def compute_rmse_between_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> tuple[float, np.ndarray]:
    """Compute RMSE between corresponding points in two point clouds.

    Calculates the Root Mean Square Error (RMSE) between a source and target
    point cloud. The point clouds must have the same number of points, as the
    distance is computed for corresponding points at the same indices.

    Args:
        source: Source point cloud.
        target: Target point cloud (must have same number of points as source).

    Returns:
        A tuple containing:
            - rmse: Root Mean Square Error (scalar).
            - distances: Per-point Euclidean distances as a (N,) array.

    Raises:
        ValueError: If the point clouds have different numbers of points.

    Note:
        This function assumes point-to-point correspondence (i.e., source.points[i]
        corresponds to target.points[i]). For registration evaluation, typically
        the source would be transformed before calling this function.
    """
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    if len(source_points) != len(target_points):
        raise ValueError(
            f"Point clouds must have the same number of points. "
            f"Source: {len(source_points)}, Target: {len(target_points)}"
        )

    dists = np.linalg.norm(source_points - target_points, axis=1)
    rmse_val = np.sqrt(np.mean(dists**2))
    logging.debug(f"Computed RMSE = {rmse_val:.6f}")
    return rmse_val, dists


def compute_rmse_transformations(
    transf_est: np.ndarray, transf_gt: np.ndarray, pcd: o3d.geometry.PointCloud
) -> float:
    """Compute the RMSE between two transformations applied to a point cloud.

    Args:
        transf_est: Estimated transformation (4x4 matrix).
        transf_gt: Ground truth transformation (4x4 matrix).
        pcd: Point cloud to which the transformations will be applied.

    Returns:
        The root mean square error (RMSE) between the point clouds obtained
        by applying T_est and T_gt to the input point cloud.
    """
    pcd_est = copy.deepcopy(pcd)
    pcd_gt = copy.deepcopy(pcd)
    pcd_est.transform(transf_est)
    pcd_gt.transform(transf_gt)
    rmse, _ = compute_rmse_between_point_clouds(pcd_est, pcd_gt)
    return rmse
