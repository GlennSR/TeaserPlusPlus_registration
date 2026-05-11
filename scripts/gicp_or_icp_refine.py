import argparse
import logging
import os
import time
from registration.utils.logging import setup_logging
import json
import scipy

import open3d as o3d
from registration.visualization.viewer import *
import numpy as np 
import copy
from helpers import *
from registration.utils.point_cloud import rough_scale_point_cloud_from_file, load_point_clouds_for_refinement
from registration.utils.metrics import calculate_errors

logger = logging.getLogger(__name__)

def refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    distance_threshold: float,
    initial_transformation: np.ndarray,
    max_iteration: int,
    use_generalized_icp: bool = False
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
    algorithm_name = "Generalized ICP (GICP)" if use_generalized_icp else "ICP"

    logger.debug(f"  Pairwise {algorithm_name} registration...")
    logger.info(
        f"to refine the alignment. This time we use a strict distance threshold {distance_threshold:.3f}"
    )

    # Select registration method
    if use_generalized_icp:
        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            distance_threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
    else:
        # Use point-to-plane ICP for better convergence
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )

    logger.debug(
        f"    Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}"
    )
    return result


def main(args: argparse.Namespace):

    # Load and visualize two point clouds
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)
    ref_voxel_size = args.refinement_voxel_size
    VISUALIZE = args.viz
    frame_size = rough_scale_point_cloud_from_file(args.target) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function

    source_raw.paint_uniform_color([1.0, 0.706, 0.0]) # show source in yellow
    target_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show target in green

    # Initiate timer
    start_time = time.time()

    # If the target point cloud is not aligned with the world frame, then we need to apply it's known transformation 
    # (Lidar to World) to align so the algorithm can estimate the transformation in the world reference frame.
    target_gt_transform_file = args.target.replace('.ply', '.json').replace('.pcd', '.json')
    try:
        with open(target_gt_transform_file, 'r') as file:
            target_toworld_transform = np.array(json.load(file)["H"])
            logger.info(f"Target Ground Truth transform: \n{target_toworld_transform}")
    except FileNotFoundError:
        logger.error(f"The file '{target_gt_transform_file}' was not found.")
    
    target_raw.transform(target_toworld_transform)

    # Load source initial guess transformation from .json file
    scan_initial_guess_filename = os.path.split(args.source)[1].replace('.ply', '.json').replace('.pcd', '.json')
    scan_initial_guess_path = os.path.join(args.estimated_poses, scan_initial_guess_filename)
    logger.info(f"Loading source initial guess transformation from: {scan_initial_guess_path}")
    try:
        with open(scan_initial_guess_path, 'r') as file:
            trans_init = np.array(json.load(file)["H"])
    except FileNotFoundError:
        logger.error(f"The file '{scan_initial_guess_path}' was not found.")
    r = scipy.spatial.transform.Rotation.from_matrix(trans_init[:3, :3])
    r = r.as_euler('xyz')
    r[2] = 0 # For the real dataset the z-axis is considered the yaw angle
    trans_init[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', r).as_matrix()
    trans_init[:3, 3] = 0
    logger.info(f"Source Initial Guess: \n{trans_init}")

    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, trans_init,
                                 window_name="Initial State (Source: Blue, Target: Red)", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4),
                                 source_frame_trans=trans_init)

    logger.info(f"Loading scan at refinement resolution ({ref_voxel_size})...")
    source_refined, target_refined = load_point_clouds_for_refinement(
    source=source_raw,
    target=target_raw,
    voxel_size=ref_voxel_size,
    trans_init=trans_init # Apply the initial estimated global transformation before the refinement
    )
        # local refinement using ICP Point to Plane
    icp_sol = refine_registration(source_refined, target_refined, ref_voxel_size*2, np.eye(4), max_iteration=args.max_iter_icp, use_generalized_icp=args.use_gicp)
    T_icp = icp_sol.transformation
    logger.info(f"Estimated transformation after refinement:\n{T_icp}")

    # visualize the registration after ICP refinement
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_icp @ trans_init, window_name="ICP Refinement", 
                                size=frame_size, 
                                target_frame_trans=np.eye(4), 
                                source_frame_trans=T_icp @ trans_init)

    # Computing elapsed time to run Teaser++ registration
    end_time = time.time()
    registration_total_time = end_time - start_time
    logger.info(f"Elapsed time for ICP/GICP Registration: {registration_total_time:.4f} seconds")

    ## METRICS ##
    # Calculate and save registration metrics 
    # Load Ground Thruth transformation from .json file
    scan_gt_json = args.source.replace('.ply', '.json').replace('.pcd', '.json')

    calculate_errors(icp_sol, T_icp @ trans_init, scan_gt_json, source_dir, registration_total_time, source_raw, target_raw)

if __name__ == "__main__":
    # tutorial from here https://teaser.readthedocs.io/en/master/quickstart.html
    
    # add input file argument
    parser = argparse.ArgumentParser(description="Teaser++ registration")
    parser.add_argument("--source", type=str, help="source file path", required=True)
    parser.add_argument("--target", type=str, help="target file path", required=True)
    parser.add_argument("--estimated-poses", type=str, help="file path to the global registration estimated poses", required=True)

    parser.add_argument(
        "--refinement-voxel-size",
        type=float,
        default=None,
        help=(
            "Voxel size (mm) for downsampling during the ICP/GICP refinement step. "
            "If not set, the Teaser voxel size is reused (no extra loading). "
            "Set to 0 to use the original undownsampled point clouds."
        ),
    )
    
    parser.add_argument(
        "--max_iter_icp", type=int, help="Input file path", default=30
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        help="Use Generalized ICP for refinement (otherwise use point-to-plane ICP)",
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
    logger.info(f"Input arguments: {input_args}")
    
    if os.path.isdir(input_args.source):
       # Create a list with only the supported point cloud files for registration
       pcl_files = [f for f in os.listdir(input_args.source) if f.endswith('.ply') or f.endswith('.pcd')]
    #    file = []
    #    for i in range(271,380):
    #         file.append(f"{i}.ply")
    #    logger.info(f"pcl_files: {file}")
       number_of_files = len(pcl_files)
    #    pcl_files = file

       logger.info(f"Source is a directory, applying TEASER++ registration to all its {number_of_files} files.")
       source_dir = copy.deepcopy(input_args.source)
       count = 1
       for filename in pcl_files:
            source_file = os.path.join(source_dir, filename)
            logger.info(f"TEASER++ registration to: {source_file}. File ({count} / {number_of_files})")
            input_args.source = source_file
            main(input_args)
            count += 1
    else:
        source_dir, _ = os.path.split(input_args.source)
        main(input_args)
