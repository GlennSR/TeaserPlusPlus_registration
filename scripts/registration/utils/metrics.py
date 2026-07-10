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
                         target_down_nb_points: int,
                         source_down_nb_points: int,
                         teaser_solver: teaserpp_python.teaserpp_python.RobustRegistrationSolver,
                         icp_sol: o3d.pipelines.registration.RegistrationResult,
                         trans_init: np.ndarray,
                         num_corrs: int,
                         NOISE_BOUND: float,
                         registration_total_time: float,
                         args: argparse.Namespace,
                         output_dir: str
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
    logger.info("\n"*2+"-"*30 + "Calculating Registration Error" + "-"*30)

    # Calculate the metric of the result transformation using Open3D compute_point_cloud_distance() method
    # Full-cloud distances
    source_raw_T_icp = copy.deepcopy(source_raw).transform(icp_sol.transformation)
    distances_o3d = target_raw.compute_point_cloud_distance(source_raw_T_icp)
    logger.debug(f"Mean Open3D distance for the registration result (full cloud): {np.mean(distances_o3d):.6f}")

    # Calculate the standard deviation of the full-cloud distances
    std_distance = np.std(distances_o3d)
    logger.debug(f"Standard deviation of distances after registration (full cloud): {std_distance:.6f}")
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
        source_raw, target_raw, NOISE_BOUND, icp_sol.transformation
    )
    logger.info(f"Open3D Evaluation Metrics:")
    logger.info(f"  Fitness: {evaluation.fitness:.4f} (fraction of source inlier points)")
    logger.info(f"  Inlier RMSE: {evaluation.inlier_rmse:.4f} mm (lower is better)")
    logger.info(f"  ICP correspondence set size: {len(evaluation.correspondence_set)} / {len(source_raw.points)} source points ({len(evaluation.correspondence_set)/len(source_raw.points)*100:.1f}%)")
    logger.info(f"  FPFH correspondences: {num_corrs}")
    logger.info("\n"*2+"-"*60 + "\n"*2)


    ## Calculate inliers mean error (distances) between the correspondent points

    # Build point clouds of the correspondent inlier points
    corr = np.asarray(evaluation.correspondence_set)
    src_corr_pts = np.asarray(source_raw.points)[corr[:,0]]
    tgt_corr_pts = np.asarray(target_raw.points)[corr[:,1]]

    logger.debug(f"Number of inlier correspondences after registration: {len(src_corr_pts)}")

    percentage_inliers_to_target = len(src_corr_pts) / len(target_raw.points)
    logger.debug(f"Percentage of inliers with respect to target point cloud: {percentage_inliers_to_target*100:.2f} %")
    percentage_inliers_to_source = len(src_corr_pts) / len(source_raw.points)
    logger.debug(f"Percentage of inliers with respect to source point cloud: {percentage_inliers_to_source*100:.2f} %")

    src_corr_pcd = o3d.geometry.PointCloud()
    tgt_corr_pcd = o3d.geometry.PointCloud()
    src_corr_pcd.points = o3d.utility.Vector3dVector(src_corr_pts)
    tgt_corr_pcd.points = o3d.utility.Vector3dVector(tgt_corr_pts)

    # Compute inliers distances after registration
    src_corr_pcd_T = copy.deepcopy(src_corr_pcd)
    src_corr_pcd_T.transform(icp_sol.transformation)
    distances_inliers = tgt_corr_pcd.compute_point_cloud_distance(src_corr_pcd_T)

    logger.debug(f"Mean distance for the registration inliers (only inliers): {np.mean(distances_inliers):.6f}")
    
    # Correct icp transformation by the initial transformation used for gravity alignment
    # Convert transaltion back to meters
    T_icp_meters = copy.deepcopy(icp_sol.transformation)
    T_icp_meters[:3, 3] /= 1000  # Convert translation from mm to m
    T_icp_corrected_meters = T_icp_meters @ trans_init

    T_icp_corrected = copy.deepcopy(T_icp_corrected_meters)
    T_icp_corrected[:3, 3] *= 1000  # Convert translation back to mm for logging

    logger.debug(f"ICP refinement result: {icp_sol}")
    logger.info(f"Estimated matrix (meters):\n{T_icp_corrected_meters}")

    # Calculate the diagonal length of the target point cloud bounding box and the RMSE as percentage of it
    max_point = np.max(np.asarray(target_raw.points), axis=0)
    min_point = np.min(np.asarray(target_raw.points), axis=0)
    target_diagonal_length = np.linalg.norm(max_point - min_point)
    logger.debug(f"Target point cloud diagonal length: {target_diagonal_length:.3f} mm")

    rmse_percentage = icp_sol.inlier_rmse / target_diagonal_length * 100
    logger.debug(f"ICP inlier RMSE as percentage of target diagonal length: {rmse_percentage:.4f} %")

    # Load Ground Thruth transformation from .json file
    source_json = args.source.replace('.ply', '.json').replace('.pcd', '.json')
    if not os.path.exists(source_json):
        source_json = args.source.replace('.ply', '_pose.json').replace('.pcd', '_pose.json')
    try:
        with open(source_json, 'r') as file:
            source_gt_transform_mm = load_gt_transform(source_json)
            logger.debug(f"Source Ground Truth transform (in mm): \n{source_gt_transform_mm}")
    except FileNotFoundError:
        logger.error(f"The file '{source_json}' was not found.")

    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    rot_err, trans_err = transformation_error(
        T_icp_corrected, source_gt_transform_mm
    )
    matrix = T_icp_corrected @ source_gt_transform_mm.T
    logger.debug(f"Product of the transformations:\n{matrix}")
    logger.debug(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f} mm"
    )

    # compute the rms error between initial and final translation (assuming that the points are corresponding)
    registration_rmse = compute_rmse_transformations(
        T_icp_corrected, source_gt_transform_mm, source_raw
    )
    logger.debug(f"Registration RMSE: {registration_rmse} mm")

    # Save the calculated metrics to a .json file
    output_metrics = {
        "source": args.source,
        "target": args.target,
        "voxel_size": args.voxel_size,
        "refinement_voxel_size": args.refinement_voxel_size,
        "rotation_error_rad": rot_err,
        "rotation_error_deg": np.degrees(rot_err),
        "translation_error": trans_err,
        "fitness": icp_sol.fitness,
        "inlier_rmse": icp_sol.inlier_rmse,
        "rmse_percentage_to_target_diagonal": rmse_percentage,
        "mean_distance_points_full_cloud": float(np.mean(distances_o3d)),
        "max_distance_points_full_cloud": float(np.max(distances_o3d)),
        "standard_deviation_distance_full_cloud": float(std_distance),
        "inlier_mean_distance": np.mean(distances_inliers),
        "registration_total_time_sec": registration_total_time,
        "nb_of_points_target_down": target_down_nb_points,
        "nb_of_points_source_down": source_down_nb_points,
        "nb_of_fpfh_correspondences": num_corrs,
        "percentage_inliers_to_target": percentage_inliers_to_target,
        "percentage_inliers_to_source": percentage_inliers_to_source,
        "estimated_transformation": T_icp_corrected.tolist(),
        "product_of_the_transformations": (T_icp_corrected @ source_gt_transform_mm).tolist()
    }

    # Print localization statistics
    logger.info("\n"*2+"-"*30 + "Localization Statistics" + "-"*30)
    logger.info(f"  Fitness: {icp_sol.fitness:.4f}")
    logger.info(f"  Inlier RMSE: {icp_sol.inlier_rmse:.4f} mm")
    logger.info(f"  RMSE as percentage of target diagonal length: {rmse_percentage:.4f} %")
    logger.info(f"Error vs Ground Truth:")
    logger.info(f"  Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f})")
    logger.info(f"  Translation error: {trans_err:.4f} mm")
    logger.info(f"Timing:")
    logger.info(f"  Registration total time: {registration_total_time:.4f} sec")
    logger.info("-"*60 + "\n"*2)

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pcd_file_path = args.source.replace('.ply', '_metrics.json').replace('.pcd', '_metrics.json')
        _, file_name = os.path.split(pcd_file_path)
        metrics_file = os.path.join(output_dir, file_name)
        with open(metrics_file, 'w') as file:
            json.dump(output_metrics, file, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
    except FileNotFoundError:
        logger.error(f"The file 'metrics.json' was not found.")
    logger.info("-"*100+"\n"*2)


# [[ 0.50636631 -0.65424811  0.56174066 -1.09790498]
#  [-0.57047087 -0.74266472 -0.35073082  4.91653141]
#  [ 0.64664995 -0.14285841 -0.74928988  1.98157184]
#  [ 0.          0.          0.          1.        ]]


def load_gt_transform(json_file):
    try:
        with open(json_file, 'r') as file:
            gt_transform = np.array(json.load(file)["H"])
            logger.info(f"Ground Truth transform (in m): \n{gt_transform}")
            gt_transform_mm = copy.deepcopy(gt_transform)
            gt_transform_mm[:3, 3] *= 1000  # Convert translation from m to mm
            return gt_transform_mm
    except FileNotFoundError:
        logger.error(f"The file '{json_file}' was not found.")
        return None


def calculate_errors(args, icp_sol, estimated_transform, voxel_size, scan_gt_json, total_time, source, target, output_dir='metrics/'):
    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    logger.info("\n"*2+"-"*30 + "Calculating Teaser++ Errors" + "-"*30)
    gt_transform = load_gt_transform(scan_gt_json)
    logger.info(f"Ground Truth transform (in mm): \n{gt_transform}")

    rot_err, trans_err = transformation_error(
        estimated_transform, gt_transform
    )
    matrix = estimated_transform @ gt_transform
    logger.info(f"Product of the transformations:\n{matrix}")
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f} mm"
    )

    if icp_sol is not None:
        fitness = icp_sol.fitness
        inlier_rmse = icp_sol.inlier_rmse
    else:
        fitness = np.nan
        inlier_rmse = np.nan

    # Full-cloud distances
    source_T_icp = copy.deepcopy(source).transform(estimated_transform)
    distances_o3d = target.compute_point_cloud_distance(source_T_icp)
    logger.info(f"Mean Open3D distance for the registration result (full cloud): {np.mean(distances_o3d):.6f}")

    # Calculate the standard deviation of the full-cloud distances
    std_distance = np.std(distances_o3d)
    logger.info(f"Standard deviation of distances after registration (full cloud): {std_distance:.6f}")

    # Save the calculated metrics to a .json file
    output_metrics = {
        "source": args.source,
        "target": args.target,
        "voxel_size": voxel_size,
        "rotation_error_rad": rot_err,
        "rotation_error_deg": np.degrees(rot_err),
        "translation_error": trans_err,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "rmse_percentage_to_target_diagonal": np.nan,
        "mean_distance_points_full_cloud": np.mean(distances_o3d),
        "max_distance_points_full_cloud": np.max(distances_o3d),
        "standard_deviation_distance_full_cloud": std_distance,
        "inlier_mean_distance": np.nan,
        "registration_total_time_sec": total_time,
        "nb_of_points_target_down": np.nan,
        "nb_of_points_source_down": np.nan,
        "nb_of_fpfh_correspondences": np.nan,
        "percentage_inliers_to_target": np.nan,
        "percentage_inliers_to_source": np.nan,
        "estimated_transformation": estimated_transform.tolist(),
        "product_of_the_transformations": matrix.tolist()
    }
    try:

        logger.info(f"Saving metrics to: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pcd_file_path = scan_gt_json.replace('.json', '_metrics.json')
        _, file_name = os.path.split(pcd_file_path)
        # metrics_file = os.path.join(source_dir, save_path, file_name)
        metrics_file = os.path.join(output_dir, file_name)
        logger.info(f"Metrics file path: {metrics_file}")
        with open(metrics_file, 'w') as file:
            json.dump(output_metrics, file, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
        logger.info("-"*100+"\n"*2)
    except FileNotFoundError:
        logger.error(f"The file '{metrics_file}' was not found.")


def save_estimated_poses(estimated_transform, source_path, output_dir):
    # Save the estimated transformation to a .json file

    output_transform = {
        "H": estimated_transform.tolist()
    }
    try:
        logger.info(f"Saving estimated poses to: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _, file_name = os.path.split(source_path)
        file_name = file_name.replace('.ply', '.json').replace('.pcd', '.json')
        poses_file = os.path.join(output_dir, file_name)
        with open(poses_file, 'w') as file:
            json.dump(output_transform, file, indent=4)
            logger.info(f"Saved poses to {poses_file}")
    except FileNotFoundError:
        logger.error(f"The file '{poses_file}' was not found.")
