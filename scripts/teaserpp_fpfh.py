import argparse
import logging
import os
import time
from registration.utils.logging import setup_logging
import json

import open3d as o3d
import matplotlib.pyplot as plt
from registration.visualization.viewer import *
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from registration.utils.point_cloud import preprocess_point_cloud, noise_Gaussian, rough_scale_point_cloud_from_file
from registration.utils.transforms import apply_random_transform
from registration.utils.solution_check import is_solution_upside_down
from registration.utils.metrics import registration_metrics

logger = logging.getLogger(__name__)

def prepare_dataset(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> tuple:
    """Load and prepare point cloud datasets for registration.
    
    Loads source and target downsampled point clouds, applies an initial transformation
    to the source cloud, and preprocesses both clouds by computing
    FPFH features for feature-based registration.

    Args:
        source: The original source point cloud.
        target: Downsampled target point cloud.
        voxel_size: The size of the voxel for downsampling both point clouds.
        trans_init: Initial transformation matrix to apply to the source cloud (default: identity matrix).
        correction: Correction transformation matrix to apply to both clouds, typically to align to the visual reference frame (default: identity matrix).

    Returns:
        A tuple containing:
            - source: The original source point cloud with initial transformation applied
            - target: The original target point cloud
            - source_down: Downsampled source point cloud
            - target_down: Downsampled target point cloud
            - source_fpfh: FPFH features of the downsampled source
            - target_fpfh: FPFH features of the downsampled target
    """
    
    source.transform(correction)
    
    target.transform(correction)

    source.transform(trans_init)

    logger.info("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(logger, source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")
    logger.info(f"Feature of SOURCE: {source_fpfh}")

    logger.info("Preprocessing target point cloud")
    target_down, target_fpfh = preprocess_point_cloud(logger, target, voxel_size)
    print_point_cloud_info(target_down, "Downsampled target")
    logger.info(f"Feature of TARGET: {target_fpfh}")

    return source, target, source_down, target_down, source_fpfh, target_fpfh

def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_threshold: float,
    initial_transformation: np.ndarray,
    max_iteration: int,
) -> o3d.pipelines.registration.RegistrationResult:
    """Refine registration using point-to-plane ICP algorithm.

    Performs Iterative Closest Point (ICP) registration with point-to-plane metric
    to refine the initial alignment obtained from global registration. This method
    uses a stricter distance threshold and operates on the original (non-downsampled)
    point clouds for higher accuracy.

    Args:
        source: Original source point cloud.
        target: Original target point cloud.
        distance_threshold: Maximum correspondence points-pair distance.
        initial_transformation: Initial transformation matrix from global registration.

    Returns:
        Registration result containing the refined transformation matrix, fitness score,
        and inlier RMSE from the point-to-plane ICP registration.
    """
    logger.info("Point-to-plane ICP registration is applied on original point clouds")
    logger.info(
        f"to refine the alignment. This time we use a strict distance threshold {distance_threshold:.3f}"
    )
    if not target.has_normals():
        logger.info("Target point cloud does not have normals, estimating them...")
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )  # @TODO check radius parameter wrt the size of the model/voxel

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    return result

def teaserpp_registration(args: argparse.Namespace):
    # Load and visualize two point clouds
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)
    noise_std = args.noise_std * 1000 # scale to match point cloud units in mm
    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz

    source_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show source in blue
    target_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show target in red
    frame_size = rough_scale_point_cloud_from_file(args.target) # scale frame size according to target to plot the axis in open3D Draw function

    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, np.eye(4), window_name="Initial State (Source: Blue, Target: Red)", size=frame_size)

    # Add a random gaussian noise chosen by the user to the source point cloud
    source_raw.points = o3d.utility.Vector3dVector(noise_Gaussian(np.asarray(source_raw.points), noise_std))

    # if VISUALIZE:
    #     draw_registration_result(source_raw, target_raw, np.eye(4), window_name="Noisy source", size=frame_size)

    # Initiate timer
    start_time = time.time()

    # idx_gravity_axis = 1 # assuming y-axis is the gravity axis
    # apply_random_transform(args.source, args.target, frame_size, idx_gravity_axis)
    
    trans_init = np.eye(4) # Don't change the source initial transformation, as we have loaded it from the ground truthfile

    source_raw, target_raw, source_down, target_down, source_feats, target_feats = prepare_dataset(source_raw, target_raw, VOXEL_SIZE, trans_init)

    if VISUALIZE:
        draw_registration_result(source_down, target_down, trans_init, window_name="Random transform on Downsampled Point Clouds", size=frame_size)

    # extract point coordinates as numpy arrays
    source_xyz = pcd2xyz(source_down) # np array of size 3 by N
    target_xyz = pcd2xyz(target_down) # np array of size 3 by M

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        source_feats, target_feats, mutual_filter=True)
    source_corr = source_xyz[:,corrs_A] # np array of size 3 by num_corrs
    target_corr = target_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = source_corr.shape[1]
    logger.info(f'FPFH generates {num_corrs} putative correspondences.')

    # visualize the point clouds together with feature correspondences
    if VISUALIZE:
        points = np.concatenate((source_corr.T,target_corr.T),axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i,i+num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([source_raw, target_raw, line_set], window_name="FPFH Correspondences")

    # TEASER++ registration
    NOISE_BOUND = VOXEL_SIZE * 2 # 2 works well
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(source_corr,target_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)
        
    # Visualize the registration results after TEASER++
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_teaser, window_name="TEASER++ Registration Results", size=frame_size)

    # local refinement using ICP Point to Plane
    icp_sol = refine_registration(source_raw, target_raw, NOISE_BOUND, T_teaser, max_iteration=args.max_iter_icp)
    # This is the estimated transformation where you can find the rotation and translation of the source in the target reference frame
    T_icp = icp_sol.transformation

    # Computing elapsed time to run Teaser++ registration
    end_time = time.time()
    registration_total_time = end_time - start_time
    logger.info(f"Elapsed time for TEASER++ Registration: {registration_total_time:.4f} seconds")

    # visualize the registration after ICP refinement
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", size=frame_size)

    ## METRICS ##
    # Calculate and save registration metrics
    registration_metrics(target_raw, source_raw, teaser_solver, icp_sol, num_corrs, NOISE_BOUND, registration_total_time, args)

if __name__ == "__main__":
    # tutorial from here https://teaser.readthedocs.io/en/master/quickstart.html
    
    # add input file argument
    parser = argparse.ArgumentParser(description="Teaser++ registration")
    parser.add_argument("--source", type=str, help="source file path", required=True)
    parser.add_argument("--target", type=str, help="target file path", required=True)

    parser.add_argument(
        "--voxel-size", type=float, help="voxels size for downsampling", default=30
    )

    parser.add_argument(
        "--noise-std", type=float, help="std deviation of gaussian noise to add to source", default=0.0
    )
    
    parser.add_argument(
        "--max_iter_icp", type=int, help="Input file path", default=2000
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--viz",
        type=bool,
        help="Visualize point clouds with open3D",
        default=False
    )

    input_args = parser.parse_args()
    # Set logging level based on user selection
    setup_logging(getattr(logging, input_args.verbose))
    
    if os.path.isdir(input_args.source):
       # Create a list with only the supported point cloud files for registration
       pcl_files = [f for f in os.listdir(input_args.source) if f.endswith('.ply') or f.endswith('.pcd')] 
       number_of_files = len(pcl_files)
       logger.info(f"Source is a directory, applying TEASER++ registration to all its {number_of_files} files.")
       source_dir = copy.deepcopy(input_args.source)
       count = 1
       for filename in pcl_files:
            source_file = os.path.join(source_dir, filename)
            logger.info(f"TEASER++ registration to: {source_file}. File ({count} / {number_of_files})")
            input_args.source = source_file
            teaserpp_registration(input_args)
            count += 1
    else:
        source_dir, _ = os.path.split(input_args.source)
        teaserpp_registration(input_args)

