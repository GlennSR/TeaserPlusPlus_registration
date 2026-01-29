"""Metrics for point cloud registration evaluation."""

import copy
import logging

import numpy as np
import open3d as o3d
import teaserpp_python
import argparse
import json
import os
import matplotlib.pyplot as plt
from registration.utils.transforms import (
    transformation_error,
)

logger = logging.getLogger(__name__)

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

def registration_metrics(target_raw: o3d.geometry.PointCloud,
                         source_raw: o3d.geometry.PointCloud,
                         teaser_solver: teaserpp_python.teaserpp_python.RobustRegistrationSolver,
                         icp_sol: o3d.pipelines.registration.RegistrationResult,
                         num_corrs: int,
                         NOISE_BOUND: float,
                         registration_total_time: float,
                         args: argparse.Namespace
                         ):
    """
    Calculate and log various metrics to evaluate the registration result.
    
    Args:
        :param target_raw: Target point cloud
        :param source_raw: Source point cloud
        :param teaser_solver: The TEASER++ solver calculated
        :param icp_sol: The ICP refinement result (Final transformation)
        :param num_corrs: Number of correspondences calculated with FPFH
        :param NOISE_BOUND: Noise bound used in TEASER++
        :param registration_total_time: Total time taken for registration
        :param args: Command line arguments
    """

    VISUALIZE = args.viz
    if os.path.isdir(args.source):
       source_dir = args.source
    else:
        source_dir, _ = os.path.split(args.source)
    # Calculate the metric of the result transformation using Open3D compute_point_cloud_distance() method
    # Full-cloud distances
    T_icp = icp_sol.transformation
    source_raw_T_icp = copy.deepcopy(source_raw).transform(T_icp)
    distances_o3d = target_raw.compute_point_cloud_distance(source_raw_T_icp)
    logger.info(f"Mean Open3D distance for the registration result (full cloud): {np.mean(distances_o3d):.6f}")

    # Calculate the standard deviation of the full-cloud distances
    std_distance = np.std(distances_o3d)
    logger.info(f"Standard deviation of distances after registration (full cloud): {std_distance:.6f}")

    if VISUALIZE:
        plt.hist(distances_o3d, bins=20, color='blue', rwidth=1.0)
        plt.title('Histogram of Point-to-Point Distances After Registration')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # TEASER++ internal quality metrics
    teaser_solution = teaser_solver.getSolution()
    translation_inliers = teaser_solver.getTranslationInliers()
    rotation_inliers = teaser_solver.getRotationInliers()
    logger.info(f"TEASER++ Internal Metrics:")
    logger.info(f"  Translation inliers: {len(translation_inliers)} / {num_corrs} ({len(translation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Rotation inliers (max clique): {len(rotation_inliers)} / {num_corrs} ({len(rotation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Solution valid: {teaser_solution.valid if hasattr(teaser_solution, 'valid') else 'N/A'}")

    # Evaluate the solution using Open3D
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_raw, target_raw, NOISE_BOUND, T_icp
    )
    logger.info(f"Open3D Evaluation Metrics:")
    logger.info(f"  Fitness: {evaluation.fitness:.4f} (fraction of inlier points)")
    logger.info(f"  Inlier RMSE: {evaluation.inlier_rmse:.4f} mm (lower is better)")
    logger.info(f"  Correspondence set size: {len(evaluation.correspondence_set)}")

    # Calculate inliers mean error (distances) between the correspondent points

    # Build point clouds of the correspondent inlier points
    corr = np.asarray(evaluation.correspondence_set)
    src_corr_pts = np.asarray(source_raw.points)[corr[:,0]]
    tgt_corr_pts = np.asarray(target_raw.points)[corr[:,1]]

    src_corr_pcd = o3d.geometry.PointCloud()
    tgt_corr_pcd = o3d.geometry.PointCloud()
    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr_pts)
    tgt_corr_pcd.points = o3d.utility.Vector3dVector(tgt_corr_pts)

    # Compute inliers distances after registration
    src_corr_pcd_T = copy.deepcopy(src_corr_pcd)
    src_corr_pcd_T.transform(T_icp)
    distances_inliers = tgt_corr_pcd.compute_point_cloud_distance(src_corr_pcd_T)

    logger.info(f"Inlier distances mean={np.mean(distances_inliers):.4f} mm")
    

    logger.info(f"ICP refinement result: {icp_sol}")
    logger.info(f"Estimated matrix:\n{icp_sol.transformation}")
    logger.info(
        f"Result fitness: {icp_sol.fitness}, inlier RMSE: {icp_sol.inlier_rmse} mm"
    )

    # Calculate the diagonal length of the target point cloud bounding box and the RMSE as percentage of it
    max_point = np.max(np.asarray(target_raw.points), axis=0)
    min_point = np.min(np.asarray(target_raw.points), axis=0)
    target_diagonal_length = np.linalg.norm(max_point - min_point)
    logger.info(f"Target point cloud diagonal length: {target_diagonal_length:.3f} mm")

    rmse_percentage = icp_sol.inlier_rmse / target_diagonal_length * 100
    logger.info(f"ICP inlier RMSE as percentage of target diagonal length: {rmse_percentage:.4f} %")

    # Load Ground Thruth transformation from .json file
    source_json = args.source.replace('.ply', '.json').replace('.pcd', '.json')
    try:
        with open(source_json, 'r') as file:
            source_gt_transform = np.array(json.load(file)["H"])
            logger.info(f"Source Ground Truth transform: \n{source_gt_transform}")
    except FileNotFoundError:
        logger.error(f"The file '{source_json}' was not found.")
    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    rot_err, trans_err = transformation_error(
        icp_sol.transformation, source_gt_transform
    )
    matrix = icp_sol.transformation @ source_gt_transform
    logger.debug(f"Product of the transformations:\n{matrix}")
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
    )

    # compute the rms error between initial and final translation (assuming that the points are corresponding)
    registration_rmse = compute_rmse_transformations(
        icp_sol.transformation, source_gt_transform, source_raw
    )
    logger.info(f"Registration RMSE: {registration_rmse}")

    # Save the calculated metrics to a .json file
    output_metrics = {
        "estimated_transformation": icp_sol.transformation.tolist(),
        "product_of_the_transformations": (T_icp @ source_gt_transform).tolist(),
        "rotation_error_rad": rot_err,
        "rotation_error_deg": np.degrees(rot_err),
        "translation_error": trans_err,
        "fitness": icp_sol.fitness,
        "inlier_rmse": icp_sol.inlier_rmse,
        "rmse_percentage_to_target_diagonal": rmse_percentage,
        "mean_distance": float(np.mean(distances_o3d)),
        "max_distance": float(np.max(distances_o3d)),
        "standard_deviation_distance": float(std_distance),
        "inlier_mean_distance": np.mean(distances_inliers),
        "registration_total_time_sec": registration_total_time
    }
    try:
        if not os.path.exists(source_dir + '/metrics'):
            os.makedirs(source_dir + '/metrics')
        pcd_file_path = args.source.replace('.ply', '_metrics.json').replace('.pcd', '_metrics.json')
        _, file_name = os.path.split(pcd_file_path)
        metrics_file = os.path.join(source_dir, 'metrics', file_name)
        with open(metrics_file, 'w') as file:
            json.dump(output_metrics, file, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
    except FileNotFoundError:
        logger.error(f"The file 'metrics.json' was not found.")