import argparse
import logging
import os
import time
from registration.utils.logging import setup_logging
import json
import scipy

import open3d as o3d
from registration.visualization.viewer import *
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from registration.utils.point_cloud import preprocess_point_cloud, rough_scale_point_cloud, rough_scale_point_cloud_from_file, load_point_clouds_files_for_refinement
from registration.utils.metrics import registration_metrics, calculate_errors, save_estimated_poses

logger = logging.getLogger(__name__)

def prepare_dataset(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    trans_init: np.ndarray = np.identity(4),
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

    source.transform(trans_init)

    logger.debug("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(logger, source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")
    logger.debug(f"Feature of SOURCE: {source_fpfh}")

    logger.debug("Preprocessing target point cloud")
    target_down, target_fpfh = preprocess_point_cloud(logger, target, voxel_size)
    print_point_cloud_info(target_down, "Downsampled target")
    logger.debug(f"Feature of TARGET: {target_fpfh}")

    return source, target, source_down, target_down, source_fpfh, target_fpfh

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
    logger.debug(
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

def teaserpp_registration(source_raw: o3d.geometry.PointCloud,
                               trans_init: np.ndarray,
        target_raw: o3d.geometry.PointCloud,
        target_toworld_transform: np.ndarray,
        VOXEL_SIZE: float,
        max_iter_icp: int,
        verbose: str,
        VISUALIZE: bool):

    frame_size = rough_scale_point_cloud(target_raw) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function
    
    if VISUALIZE:
        mesh_frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0]
        )

        mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size*1.3, origin=[0, 0, 0]
    )

        mesh_frame_target.transform(target_toworld_transform)
        
        o3d.visualization.draw_geometries(  # type: ignore
            [target_raw, mesh_frame_target, mesh_frame_world], window_name="Target"
        )
        draw_registration_result(source_raw, target_raw, np.eye(4),
                                 window_name="Initial State (Source: Blue, Target: Red)", 
                                 size=frame_size, 
                                 target_frame_trans=target_toworld_transform)


    # logger.debug(f"axis aligned:\n{trans_init @ np.eye(4)[:, idx_gravity_axis]}")
    logger.debug(f"Updated initial transformation:\n{trans_init}")


    if VISUALIZE:
        draw_registration_result(
            source_raw, target_raw, trans_init, "Corrected settings", 
            size=frame_size, 
            target_frame_trans=target_toworld_transform, 
            source_frame_trans=trans_init
        )

    source_raw, target_raw, source_down, target_down, source_feats, target_feats = prepare_dataset(source_raw, target_raw, VOXEL_SIZE, trans_init)
    source_down_nb_points = len(source_down.points)
    target_down_nb_points = len(target_down.points)

    if VISUALIZE:
        draw_registration_result(source_down, target_down, np.eye(4), 
                                 window_name="Downsampled Point Clouds", 
                                 size=frame_size)

    # extract point coordinates as numpy arrays
    source_xyz = pcd2xyz(source_down) # np array of size 3 by N
    target_xyz = pcd2xyz(target_down) # np array of size 3 by M

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        source_feats, target_feats, mutual_filter=True)
    source_corr = source_xyz[:,corrs_A] # np array of size 3 by num_corrs
    target_corr = target_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = source_corr.shape[1]
    logger.debug(f'FPFH generates {num_corrs} putative correspondences.')

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
    NOISE_BOUND = VOXEL_SIZE * 6
    teaser_solver = get_teaser_solver_test(NOISE_BOUND)
    teaser_solver.solve(source_corr,target_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)

    # Visualize the registration results after TEASER++
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_teaser, window_name="TEASER++ Registration Results", 
                                 size=frame_size, 
                                 target_frame_trans=target_toworld_transform, 
                                 source_frame_trans=T_teaser @ trans_init)

    return T_teaser, teaser_solver, target_down_nb_points, source_down_nb_points, num_corrs
    

def main(args: argparse.Namespace):

    # Load and visualize two point clouds
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)

    # The point clouds are entered in meters, so we scale them to millimiters
    source_raw.scale(1000, center=(0, 0, 0)) # scale to mm
    target_raw.scale(1000, center=(0, 0, 0)) # scale to mm

    # Calculate the diagonal length of the target point cloud bounding box and the RMSE as percentage of it
    max_point = np.max(np.asarray(target_raw.points), axis=0)
    min_point = np.min(np.asarray(target_raw.points), axis=0)
    target_diagonal_length = np.linalg.norm(max_point - min_point)
    logger.info(f"Target point cloud diagonal length: {target_diagonal_length:.3f} mm")

    # Calculate the diagonal length of the source point cloud bounding box and the RMSE as percentage of it
    max_point = np.max(np.asarray(source_raw.points), axis=0)
    min_point = np.min(np.asarray(source_raw.points), axis=0)
    source_diagonal_length = np.linalg.norm(max_point - min_point)
    logger.info(f"Source point cloud diagonal length: {source_diagonal_length:.3f} mm")

    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz
    frame_size = rough_scale_point_cloud_from_file(args.target) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function

    source_raw.paint_uniform_color([1.0, 0.706, 0.0]) # show source in yellow
    target_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show target in blue

    # Initiate timer
    start_time = time.time()
    pose_sufix = '.json'
    # Load source initial guess transformation from .json file
    source_initial_guess_file = args.source.replace('.ply', pose_sufix).replace('.pcd', pose_sufix)
    try:
        with open(source_initial_guess_file, 'r') as file:
            trans_init = np.array(json.load(file)["H"])
    except FileNotFoundError:
        logger.error(f"The file '{source_initial_guess_file}' was not found.")
    
    # In order to simulate a initial guess transformation the code gets only the ground truth rotations on x and y axis to align
    # the source to the same vertical axis Z of the target so the point clouds are parallel.
    r = scipy.spatial.transform.Rotation.from_matrix(trans_init[:3, :3])
    r = r.as_euler('xyz')
    r[2] = 0 # For the real dataset the z-axis is considered the yaw angle
    trans_init[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', r).as_matrix()
    trans_init[:3, 3] = 0
    # logger.debug(f"Source Initial Guess: \n{trans_init}")

    # If the target point cloud is not aligned with the world frame, then we need to apply it's known transformation 
    # (Lidar to World) to align so the algorithm can estimate the transformation in the world reference frame.
    target_gt_transform_file = args.target.replace('.ply', '_gt_transform.json').replace('.pcd', '_gt_transform.json')
    if not os.path.exists(target_gt_transform_file):
        target_gt_transform_file = args.target.replace('.ply', pose_sufix).replace('.pcd', pose_sufix)
    
    try:
        with open(target_gt_transform_file, 'r') as file:
            target_toworld_transform = np.array(json.load(file)["H"])
            logger.debug(f"Target Ground Truth transform: \n{target_toworld_transform}")
    except:
        logger.warning(f"The file '{target_gt_transform_file}' was not found, assuming identity transform for target point cloud.")
        target_toworld_transform = np.eye(4)
    
    if not os.path.exists(target_gt_transform_file):
        logger.warning(f"No ground truth transform file found for target point cloud. Assuming identity transform.")
        target_toworld_transform = np.eye(4)
    
    target_raw.transform(target_toworld_transform) 

    T_teaser, teaser_solver, target_down_nb_points, source_down_nb_points, num_corrs = teaserpp_registration(source_raw, trans_init, 
                                                             target_raw, target_toworld_transform, 
                                                             VOXEL_SIZE, args.max_iter_icp, args.verbose, VISUALIZE)
    
    logger.info("\n"*2+f"Estimated Teaser++ transformation (mm):\n{T_teaser @ trans_init}"+"\n"*2)

    # Convert transaltion back to meters
    T_teaser_meters = copy.deepcopy(T_teaser)
    T_teaser_meters[:3, 3] /= 1000  # Convert translation from mm to m

    teaser_reg_time = time.time() - start_time

    if args.output is not None:
        # Save the estimated Teaser++ transformation
        output_estimated_poses_dir = os.path.join(args.output, f'Voxel{int(VOXEL_SIZE)}/Teaser_estimated_poses_{int(VOXEL_SIZE)}/')
        save_estimated_poses(T_teaser_meters @ trans_init, args.source, output_estimated_poses_dir)

        # Load Ground Thruth transformation from .json file
        scan_gt_json = source_initial_guess_file
        teaser_metrics_dir = os.path.join(args.output, f'Voxel{int(VOXEL_SIZE)}/Teaser_metrics_{int(VOXEL_SIZE)}/')
        # Calculate Teaser++ errors
        logger.info(f"T_teaser: \n{T_teaser}")
        logger.info(f"trans_init: \n{trans_init}")
        calculate_errors(args, None, T_teaser @ trans_init, VOXEL_SIZE, scan_gt_json, teaser_reg_time, source_raw, target_raw, teaser_metrics_dir)

    if args.refine_registration:
        ref_voxel_size = args.refinement_voxel_size if args.refinement_voxel_size is not None else VOXEL_SIZE
        logger.debug(f"Loading scan at refinement resolution ({ref_voxel_size})...")
        source_refined, target_refined = load_point_clouds_files_for_refinement(
        source_ply=args.source,
        target_ply=args.target,
        voxel_size=ref_voxel_size,
        trans_init=trans_init
        )
         # local refinement using ICP Point to Plane
        icp_sol = refine_registration(source_refined, target_refined, ref_voxel_size * 2, T_teaser, max_iteration=args.max_iter_icp, use_generalized_icp=args.use_gicp)
        T_icp = icp_sol.transformation
        logger.debug(f"Estimated transformation after refinement:\n{T_icp}")

        # visualize the registration after ICP refinement
        if VISUALIZE:
            draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", 
                                    size=1, 
                                    target_frame_trans=target_toworld_transform, 
                                    source_frame_trans=T_icp @ trans_init)

        # Computing elapsed time to run Teaser++ registration
        end_time = time.time()
        registration_total_time = end_time - start_time
        logger.debug(f"Elapsed time for TEASER++ Registration: {registration_total_time:.4f} seconds")

        ## METRICS ##
        # Calculate and save registration metrics 
        if args.output is not None:
            registration_metrics_dir = os.path.join(args.output, f'Voxel{int(args.voxel_size)}/Ref_Voxel{int(args.refinement_voxel_size)}')
            registration_metrics(target_raw, source_raw, target_down_nb_points, source_down_nb_points, teaser_solver, icp_sol, trans_init, num_corrs, VOXEL_SIZE * 2, registration_total_time, args, registration_metrics_dir)

            # Convert transaltion back to meters
            T_icp_meters = copy.deepcopy(T_icp)
            T_icp_meters[:3, 3] /= 1000  # Convert translation from mm to m

            estimated_poses_dir = os.path.join(args.output, f'Voxel{int(args.voxel_size)}/Ref_Voxel{int(args.refinement_voxel_size)}/Estimated_Poses/')
            save_estimated_poses(T_icp_meters @ trans_init, args.source, estimated_poses_dir) # Save poses in meters


if __name__ == "__main__":
    # tutorial from here https://teaser.readthedocs.io/en/master/quickstart.html
    
    # add input file argument
    parser = argparse.ArgumentParser(description="Teaser++ registration")
    parser.add_argument("--source", type=str, help="source file path", required=True)
    parser.add_argument("--target", type=str, help="target file path", required=True)

    parser.add_argument(
        "--start-index", type=int, help="start index for processing files in a directory", default=0
    )

    parser.add_argument(
        "--end-index", type=int, help="end index for processing files in a directory", default=None
    )

    parser.add_argument(
        "--voxel-size", type=float, help="voxels size for downsampling", default=30
    )

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
        "--noise-std", type=float, help="std deviation of gaussian noise to add to source", default=0.0
    )

    parser.add_argument(
        "--translation", type=lambda s: np.array([float(item) for item in s.split(',')]), help="Translation vector to apply to both source and target point clouds (in mm)", default="0,0,0"
    )
    
    parser.add_argument(
        "--max_iter_icp", type=int, help="Input file path", default=30
    )

    parser.add_argument(
        "--refine-registration",
        action="store_true",
        help="Refine poses after TEASER++ registration using GICP or ICP",
    )

    parser.add_argument(
        "--use-gicp",
        action="store_true",
        help="Use Generalized ICP for refinement (otherwise use point-to-plane ICP)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON folder to save detailed results",
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
    if input_args.output is not None:
        if not os.path.exists(input_args.output):
            os.makedirs(input_args.output)
        setup_logging(getattr(logging, input_args.verbose), filename=input_args.output + f"/Voxel{int(input_args.voxel_size)}.log", filemode="w")
    else:
        setup_logging(getattr(logging, input_args.verbose))
    logger.info(f"Input arguments: {input_args}")
    
    if os.path.isdir(input_args.source):
        # Create a list with only the supported point cloud files for registration
        pcl_files = [f for f in os.listdir(input_args.source) if f.endswith('.ply') or f.endswith('.pcd')]
        number_of_files = len(pcl_files)
        pcl_files.sort(key=lambda x: int(x.split('.')[0])) # sort files numerically to ensure consistent order
        for f in pcl_files:
           guess_file = f.replace('.ply', '.json').replace('.pcd', '.json')
           if not os.path.exists(os.path.join(input_args.source, guess_file)):
               raise FileNotFoundError(f"Missing JSON file for {f}.")
           
        if input_args.start_index < 0 or input_args.start_index >= number_of_files:
            raise ValueError(f"Start index {input_args.start_index} is out of bounds for the number of files ({number_of_files}).")

        if input_args.end_index is not None:
            if input_args.end_index < 0 or input_args.end_index > number_of_files:
                raise ValueError(f"End index {input_args.end_index} is out of bounds for the number of files ({number_of_files}).")
            end = input_args.end_index
        elif input_args.end_index is None:
            end = number_of_files


        logger.debug(f"Source is a directory, applying TEASER++ registration from file {input_args.start_index} to {end}.")
        source_dir = copy.deepcopy(input_args.source)
        count = 1
        for filename in pcl_files[input_args.start_index:end]:
            if (count-1)%10 == 0:
                logger.info(f"Skipping file {filename}")
            else:
                source_file = os.path.join(source_dir, filename)
                logger.info(f"TEASER++ registration to: {source_file}. File ({count} / {number_of_files})")
                input_args.source = source_file
                main(input_args)
            count += 1
    else:
        source_dir, _ = os.path.split(input_args.source)
        main(input_args)
