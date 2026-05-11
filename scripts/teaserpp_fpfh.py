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
from registration.utils.point_cloud import preprocess_point_cloud, noise_Gaussian, rough_scale_point_cloud, rough_scale_point_cloud_from_file, align_centers, load_point_clouds_for_refinement, load_point_clouds_files_for_refinement, filter_points_far_from_center
from registration.utils.transforms import apply_random_transform, generate_random_rotation_matrix, gravity_transformation, transformation_error
from registration.utils.metrics import registration_metrics, calculate_errors, save_reg_poses

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

    logger.info("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(logger, source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")
    logger.debug(f"Feature of SOURCE: {source_fpfh}")

    logger.info("Preprocessing target point cloud")
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


def translate_point_clouds(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, translation: np.ndarray):
    """Translate both source and target point clouds by a specified translation vector."""
    source.translate(translation)
    target.translate(translation)


def distance_between_points(pcd: o3d.geometry.PointCloud, name: str = "Point Cloud"):
    """Compute distance statistics between each point and its nearest neighbor.
    
    For each point in the point cloud, finds the nearest neighbor (excluding itself)
    and computes the Euclidean distance. Logs min, max, mean, median, and std.
    
    Args:+
        pcd: The point cloud to analyze.
        name: Name of the point cloud for logging.
    
    Returns:
        distances: np.ndarray of nearest-neighbor distances for each point.
    """
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    if num_points < 2:
        logger.warning(f"Point cloud '{name}' has fewer than 2 points, cannot compute distances.")
        return np.array([])
    
    # Build KD-tree and query k=2 (first neighbor is the point itself at distance 0)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = np.zeros(num_points)
    
    for i in range(num_points):
        [k, idx, dist2] = pcd_tree.search_knn_vector_3d(points[i], 2)
        # idx[0] is the point itself (distance = 0), idx[1] is the nearest neighbor
        distances[i] = np.sqrt(dist2[1])
    
    logger.info(f"--- Nearest-neighbor distance stats for '{name}' ({num_points} points) ---")
    logger.info(f"  Min distance:    {distances.min():.4f} mm")
    logger.info(f"  Max distance:    {distances.max():.4f} mm")
    logger.info(f"  Mean distance:   {distances.mean():.4f} mm")
    logger.info(f"  Median distance: {np.median(distances):.4f} mm")
    logger.info(f"  Std distance:    {distances.std():.4f} mm")
    logger.info(f"  Total points:    {num_points}")
    
    # Histogram of distances in buckets
    percentiles = np.percentile(distances, [5, 25, 50, 75, 95])
    logger.info(f"  Percentiles (5/25/50/75/95): "
                f"{percentiles[0]:.2f} / {percentiles[1]:.2f} / {percentiles[2]:.2f} / "
                f"{percentiles[3]:.2f} / {percentiles[4]:.2f} mm")
    logger.info(f"---------------------------------------------------------------")
    
    return distances


def print_points_per_voxel(pcd, voxel_size):
    """Print statistics about how many points fall into each voxel.
    
    Assigns each point to a voxel index and counts occupancy per voxel.
    """
    points = np.asarray(pcd.points)
    # Compute voxel indices for each point (same logic as Open3D's voxel_down_sample)
    voxel_min = points.min(axis=0)
    voxel_indices = ((points - voxel_min) / voxel_size).astype(int)
    # Convert 3D indices to unique keys
    _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    
    logger.info(f"--- Voxel occupancy stats (voxel_size={voxel_size}) ---")
    logger.info(f"  Total points:  {len(points)}")
    logger.info(f"  Total voxels:  {len(counts)}")
    logger.info(f"  Min points/voxel:  {counts.min()}")
    logger.info(f"  Max points/voxel:  {counts.max()}")
    logger.info(f"  Mean points/voxel: {counts.mean():.2f}")
    logger.info(f"  Median points/voxel: {np.median(counts):.1f}")
    logger.info(f"  Std points/voxel:  {counts.std():.2f}")
    
    # Histogram: how many voxels have 1 point, 2 points, etc.
    max_bin = min(counts.max(), 50)  # cap at 50 for readability
    for n in range(1, max_bin + 1):
        num_voxels = np.sum(counts == n)
        if num_voxels > 0:
            logger.info(f"  Voxels with {n:>3d} point(s): {num_voxels}")
    if counts.max() > max_bin:
        num_voxels = np.sum(counts > max_bin)
        logger.info(f"  Voxels with >{max_bin:>3d} point(s): {num_voxels}")
    logger.info(f"-----------------------------------------------")


def teaserpp_registration(source_raw: o3d.geometry.PointCloud,
                               trans_init: np.ndarray,
        target_raw: o3d.geometry.PointCloud,
        target_toworld_transform: np.ndarray,
        VOXEL_SIZE: float,
        max_iter_icp: int,
        verbose: str,
        VISUALIZE: bool):
    
    # Print point per voxel for the source pcd
    # print_points_per_voxel(source_raw, VOXEL_SIZE)

    frame_size = rough_scale_point_cloud(target_raw) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function
    
    if VISUALIZE:
        mesh_frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=500, origin=[0, 0, 0]
        )

        mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1200, origin=[0, 0, 0]
    )

        mesh_frame_target.transform(target_toworld_transform)
        
        o3d.visualization.draw_geometries(  # type: ignore
            [target_raw, mesh_frame_target, mesh_frame_world], window_name="Target"
        )
        draw_registration_result(source_raw, target_raw, np.eye(4),
                                 window_name="Initial State (Source: Blue, Target: Red)", 
                                 size=frame_size, 
                                 target_frame_trans=target_toworld_transform)
    
    logger.info(f"Updated initial transformation:\n{trans_init}")

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
    NOISE_BOUND = VOXEL_SIZE * 2
    teaser_solver = get_teaser_solver(NOISE_BOUND)
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

    # source_raw.scale(0.001, center=(0, 0, 0)) # scale to m

    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz
    frame_size = rough_scale_point_cloud_from_file(args.target) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function

    source_raw.paint_uniform_color([1.0, 0.706, 0.0]) # show source in yellow
    target_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show target in green

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
    logger.info(f"Source Initial Guess: \n{trans_init}")

    rot_180 = scipy.spatial.transform.Rotation.from_euler('z', 180, degrees=True).as_matrix()
    logger.info(f"rotation 180 around z-axis:\n{rot_180}")
    trans_init[:3, :3] = rot_180 @ trans_init[:3, :3]

    # trans_init = np.eye(4) # Loads the source without any initial guess to test the robustness of the algorithm for this case

    # If the target point cloud is not aligned with the world frame, then we need to apply it's known transformation 
    # (Lidar to World) to align so the algorithm can estimate the transformation in the world reference frame.
    target_gt_transform_file = args.target.replace('.ply', '_gt_transform.json').replace('.pcd', '_gt_transform.json')
    if not os.path.exists(target_gt_transform_file):
        target_gt_transform_file = args.target.replace('.ply', pose_sufix).replace('.pcd', pose_sufix)
    try:
        with open(target_gt_transform_file, 'r') as file:
            target_toworld_transform = np.array(json.load(file)["H"])
            logger.info(f"Target Ground Truth transform: \n{target_toworld_transform}")
    except FileNotFoundError:
        logger.error(f"The file '{target_gt_transform_file}' was not found.")
    
    target_raw.transform(target_toworld_transform) 

    # Filter out source points that are too far from its center of mass to speed up the registration and avoid outliers
    source_raw = filter_points_far_from_center(source_raw, 12)

    T_teaser, teaser_solver, target_down_nb_points, source_down_nb_points, num_corrs = teaserpp_registration(source_raw, trans_init, 
                                                             target_raw, target_toworld_transform, 
                                                             VOXEL_SIZE, args.max_iter_icp, args.verbose, VISUALIZE)
    logger.info(f"Estimated transformation:\n{T_teaser @ trans_init}")

    save_reg_poses(T_teaser @ trans_init, args.source, f'teaser_estimated_poses_{int(float(VOXEL_SIZE)*1000)}/')

    # Load Ground Thruth transformation from .json file
    scan_gt_json = source_initial_guess_file

    teaser_reg_time = time.time() - start_time
    calculate_errors(None, T_teaser @ trans_init, scan_gt_json, source_dir, teaser_reg_time, source_raw, target_raw, f'/teaser_metrics_{int(float(VOXEL_SIZE)*1000)}/')

    if args.refine_registration:
        ref_voxel_size = args.refinement_voxel_size if args.refinement_voxel_size is not None else VOXEL_SIZE
        logger.info(f"Loading scan at refinement resolution ({ref_voxel_size})...")
        source_refined, target_refined = load_point_clouds_files_for_refinement(
        source_ply=args.source,
        target_ply=args.target,
        voxel_size=ref_voxel_size,
        trans_init=trans_init
        )
         # local refinement using ICP Point to Plane
        icp_sol = refine_registration(source_refined, target_refined, ref_voxel_size*2, T_teaser, max_iteration=args.max_iter_icp, use_generalized_icp=args.use_gicp)
        T_icp = icp_sol.transformation
        logger.info(f"Estimated transformation after refinement:\n{T_icp}")

        # visualize the registration after ICP refinement
        if VISUALIZE:
            draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", 
                                    size=1, 
                                    target_frame_trans=target_toworld_transform, 
                                    source_frame_trans=T_icp @ trans_init)

        # Computing elapsed time to run Teaser++ registration
        end_time = time.time()
        registration_total_time = end_time - start_time
        logger.info(f"Elapsed time for TEASER++ Registration: {registration_total_time:.4f} seconds")

        ## METRICS ##
        # Calculate and save registration metrics 
        registration_metrics(target_raw, source_raw, target_down_nb_points, source_down_nb_points, teaser_solver, icp_sol, trans_init, num_corrs, VOXEL_SIZE * 2, registration_total_time, args)


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
    #    for i in range(0,28):
    #         file.append(f"{i}.ply")
    #    logger.info(f"pcl_files: {file}")
    #    pcl_files = file
       number_of_files = len(pcl_files)

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
