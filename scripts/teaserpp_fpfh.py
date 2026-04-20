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
from registration.utils.point_cloud import preprocess_point_cloud, noise_Gaussian, rough_scale_point_cloud, rough_scale_point_cloud_from_file, align_centers_from_files, align_centers
from registration.utils.transforms import apply_random_transform, generate_random_rotation_matrix, gravity_transformation, transformation_error
from registration.utils.solution_check import is_solution_upside_down
from registration.utils.metrics import registration_metrics

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

def translate_point_clouds(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, translation: np.ndarray):
    """Translate both source and target point clouds by a specified translation vector."""
    source.translate(translation)
    target.translate(translation)


def teaserpp_registration(
        source_raw: o3d.geometry.PointCloud,
        initial_guess_transform: np.ndarray,
        target_raw: o3d.geometry.PointCloud,
        voxel_size: float,
        max_iter_icp: int,
        verbose: str,
        visualize: bool
):
    setup_logging(getattr(logging, verbose))
    frame_size = rough_scale_point_cloud(target_raw) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function

    # Ask to Mathis if the target will already be aligned with the world frame before calling Teaser++
    # ------------------------------------------------------------------------------------------------------------------
    # target_gt_transform_file = args.target.replace('.ply', '_gt_transform.json').replace('.pcd', '_gt_transform.json')
    # try:
    #     with open(target_gt_transform_file, 'r') as file:
    #         target_toworld_transform = np.array(json.load(file)["H"])
    #         logger.info(f"Target Ground Truth transform: \n{target_toworld_transform}")
    # except FileNotFoundError:
    #     logger.error(f"The file '{target_gt_transform_file}' was not found.")
    
    # target_raw.transform(target_toworld_transform) 
    # ------------------------------------------------------------------------------------------------------------------

    # Initiate timer
    start_time = time.time()

    trans_init = initial_guess_transform
    logger.info(f"Source Initial Guess: \n{trans_init}")

    if visualize:

        mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=900, origin=[0, 0, 0]
        )
        
        o3d.visualization.draw_geometries(  # type: ignore
            [target_raw, mesh_frame_world], window_name="Target"
        )
        if visualize:
            draw_registration_result(
                source_raw, target_raw, trans_init, "Initial State (Source: Yellow, Target: Blue)", 
                size=frame_size, 
                target_frame_trans=np.eye(4), 
                source_frame_trans=trans_init
            )
    
    # ---------- TEMPORARY: Applying a random transformation for testing purpose only ----------------
    trans_init = np.asarray(
        [
            [0.862, 0.011, -0.507, 3.10005 * frame_size],
            [-0.139, 0.967, -0.215, 3.51007 * frame_size],
            [0.487, 0.255, 0.835, -0.4 * frame_size],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    rotation = [[-0.28891384, -0.68714811, -0.66660053],
                [ 0.78539809, -0.56827341,  0.24538778],
                [-0.54742911, -0.45265086,  0.70386687]]

    # rotation = generate_random_rotation_matrix()

    # logger.info(f"Applying a random rotation to the initial transformation to simulate a more realistic initial misalignment:\n{rotation}")
    trans_init[:3, :3] = rotation
    # trans_init = np.eye(4)

    # supposing that we know an estimation of the gravity vector (e.g. along the y-axis/up vector)
    # we can try to use it to align the point clouds so that y-axis is aligned
    # here we use the y vector of the initial transformation and perturb it a bit to simulate the
    # direction of the gravity
    idx_gravity_axis = 2

    gravity_transform = gravity_transformation(
        trans_init[:3, idx_gravity_axis], gravity_axis=idx_gravity_axis
    )
    trans_init = gravity_transform @ trans_init

    trans_init = (
        align_centers(source_raw, target_raw, trans_init, np.eye(4))
        @ trans_init
    )
    # ----------------------------------------------------------------------------------------
    

    # logger.debug(f"axis aligned:\n{trans_init @ np.eye(4)[:, idx_gravity_axis]}")
    # logger.info(f"Updated initial transformation:\n{trans_init}")

    if visualize:
        draw_registration_result(
            source_raw, target_raw, trans_init, "Corrected settings", 
            size=frame_size, 
            target_frame_trans=np.eye(4), 
            source_frame_trans=trans_init
        )

    source_raw, target_raw, source_down, target_down, source_feats, target_feats = prepare_dataset(source_raw, target_raw, voxel_size, trans_init)

    if visualize:
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
    if visualize:
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
    NOISE_BOUND = voxel_size * 2
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(source_corr,target_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)

    # --------------------------------------------------------------------------------------------------
    # Load Ground Thruth transformation from .json file
    source_gt_transform = initial_guess_transform
    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    rot_err, trans_err = transformation_error(
        T_teaser @ trans_init, source_gt_transform
    )
    matrix = T_teaser @ source_gt_transform
    logger.debug(f"Product of the transformations:\n{matrix}")
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
    )
    # --------------------------------------------------------------------------------------------------
       
        
    # Visualize the registration results after TEASER++
    if visualize:
        draw_registration_result(source_raw, target_raw, T_teaser, window_name="TEASER++ Registration Results", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4), 
                                 source_frame_trans=T_teaser @ trans_init)

    # local refinement using ICP Point to Plane
    icp_sol = refine_registration(source_raw, target_raw, NOISE_BOUND, T_teaser, max_iteration=max_iter_icp)
    # This is the estimated transformation where you can find the rotation and translation of the source in the target reference frame
    T_icp = icp_sol.transformation 

    # Computing elapsed time to run Teaser++ registration
    end_time = time.time()
    registration_total_time = end_time - start_time
    logger.info(f"Elapsed time for TEASER++ Registration: {registration_total_time:.4f} seconds")

    # visualize the registration after ICP refinement
    if visualize:
        draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4), 
                                 source_frame_trans=T_icp @ trans_init)
        
    # Correct Estimated matrix by the trans_init matrix used for alignment with the gravity
    estimated_transform = icp_sol.transformation @ trans_init

    return estimated_transform, icp_sol, teaser_solver, num_corrs, len(target_down.points), len(source_down.points),  registration_total_time, trans_init


def distance_between_points(pcd: o3d.geometry.PointCloud, name: str = "Point Cloud"):
    """Compute distance statistics between each point and its nearest neighbor.
    
    For each point in the point cloud, finds the nearest neighbor (excluding itself)
    and computes the Euclidean distance. Logs min, max, mean, median, and std.
    
    Args:
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


def teaserpp_registration_simulated(args: argparse.Namespace):
    # Load and visualize two point clouds
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)
    noise_std = args.noise_std * 1000 # scale to match point cloud units in mm
    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz

    source_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show source in blue
    target_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show target in red
    frame_size = rough_scale_point_cloud_from_file(args.target) # scale frame size according to target to plot the axis in open3D Draw function

    # for i in range(target_raw.points.__len__()):
    #     logger.info(f"Target point {i}: {target_raw.points[i]}")
    distance_between_points(source_raw, "Source (raw)")
    # distance_between_points(target_raw, "Target (raw)")

    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, np.eye(4), window_name="Initial State (Source: Blue, Target: Red)", size=frame_size)

    # Add a random gaussian noise chosen by the user to the source point cloud
    source_raw.points = o3d.utility.Vector3dVector(noise_Gaussian(np.asarray(source_raw.points), noise_std))

    # if VISUALIZE:
    #     draw_registration_result(source_raw, target_raw, np.eye(4), window_name="Noisy source", size=frame_size)

    # Initiate timer
    start_time = time.time()

    source_initial_guess_file = args.source.replace('.ply', '.json').replace('.pcd', '.json')
    try:
        with open(source_initial_guess_file, 'r') as file:
            trans_init = np.array(json.load(file)["H"])
    except FileNotFoundError:
        logger.error(f"The file '{source_initial_guess_file}' was not found.")
    r = scipy.spatial.transform.Rotation.from_matrix(trans_init[:3, :3])
    r = r.as_euler('xyz')
    r[1] = 0 # For the simulated dataset the y-axis is considered the yaw angle
    trans_init[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', r).as_matrix()
    trans_init[:3, 3] = 0
    logger.info(f"Source Initial Guess: \n{trans_init}")

    # trans_init = (
    #     align_centers_from_files(args.source, args.target, trans_init, np.eye(4))
    #     @ trans_init
    # )

    # If the target point cloud is not aligned with the world frame, then we need to apply it's known transformation 
    # (Lidar to World) to align so the algorithm can estimate the transformation in the world reference frame.
    target_gt_transform_file = args.target.replace('.ply', '_gt_transform.json').replace('.pcd', '_gt_transform.json')
    try:
        with open(target_gt_transform_file, 'r') as file:
            target_toworld_transform = np.array(json.load(file)["H"])
            logger.info(f"Target Ground Truth transform: \n{target_toworld_transform}")
    except FileNotFoundError:
        logger.error(f"The file '{target_gt_transform_file}' was not found.")
    
    target_raw.transform(target_toworld_transform)

    if VISUALIZE:
        draw_registration_result(
            source_raw, target_raw, trans_init, "Corrected settings", 
            size=frame_size, 
            target_frame_trans=target_toworld_transform, 
            source_frame_trans=trans_init
        )
    source_raw, target_raw, source_down, target_down, source_feats, target_feats = prepare_dataset(source_raw, target_raw, VOXEL_SIZE, trans_init)

    # if VISUALIZE:
    #     draw_registration_result(source_down, target_down, np.eye(4), window_name="Random transform on Downsampled Point Clouds", size=frame_size)

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
    registration_metrics(target_raw, source_raw, len(target_down.points), len(source_down.points), teaser_solver, icp_sol, trans_init, num_corrs, NOISE_BOUND, registration_total_time, args)


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


def teaserpp_registration_real(args: argparse.Namespace):
    # Load and visualize two point clouds
    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)
    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz

    source_raw.paint_uniform_color([1.0, 0.706, 0.0]) # show source in yellow
    target_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show target in green
    frame_size = rough_scale_point_cloud_from_file(args.target) / 7.5 # scale frame size according to target to plot the axis in open3D Draw function

    # Print point per voxel for the source pcd
    # print_points_per_voxel(source_raw, VOXEL_SIZE)

    # If the target point cloud is not aligned with the world frame, then we need to apply it's known transformation 
    # (Lidar to World) to align so the algorithm can estimate the transformation in the world reference frame.

    # target_gt_transform_file = args.target.replace('.ply', '_gt_transform.json').replace('.pcd', '_gt_transform.json')
    # try:
    #     with open(target_gt_transform_file, 'r') as file:
    #         target_toworld_transform = np.array(json.load(file)["H"])
    #         logger.info(f"Target Ground Truth transform: \n{target_toworld_transform}")
    # except FileNotFoundError:
    #     logger.error(f"The file '{target_gt_transform_file}' was not found.")
    
    # target_raw.transform(target_toworld_transform) 
    
    if VISUALIZE:
        mesh_frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=500, origin=[0, 0, 0]
        )

        mesh_frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1200, origin=[0, 0, 0]
    )

        # mesh_frame_target.transform(target_toworld_transform)
        
        o3d.visualization.draw_geometries(  # type: ignore
            [target_raw, mesh_frame_target, mesh_frame_world], window_name="Target"
        )
        draw_registration_result(source_raw, target_raw, np.eye(4),
                                 window_name="Initial State (Source: Blue, Target: Red)", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4),)

    # Initiate timer
    start_time = time.time()
    
    source_initial_guess_file = args.source.replace('.ply', '.json').replace('.pcd', '.json')
    try:
        with open(source_initial_guess_file, 'r') as file:
            trans_init = np.array(json.load(file)["H"])
    except FileNotFoundError:
        logger.error(f"The file '{source_initial_guess_file}' was not found.")
    r = scipy.spatial.transform.Rotation.from_matrix(trans_init[:3, :3])
    r = r.as_euler('xyz')
    r[2] = 0 # For the real dataset the z-axis is considered the yaw angle
    trans_init[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', r).as_matrix()
    trans_init[:3, 3] = 0
    logger.info(f"Source Initial Guess: \n{trans_init}")

    trans_init = np.asarray(
        [
            [0.862, 0.011, -0.507, 3.10005 * frame_size],
            [-0.139, 0.967, -0.215, 3.51007 * frame_size],
            [0.487, 0.255, 0.835, -0.4 * frame_size],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    rotation = [[-0.28891384, -0.68714811, -0.66660053],
                [ 0.78539809, -0.56827341,  0.24538778],
                [-0.54742911, -0.45265086,  0.70386687]]

    # rotation = generate_random_rotation_matrix()

    # logger.info(f"Applying a random rotation to the initial transformation to simulate a more realistic initial misalignment:\n{rotation}")
    trans_init[:3, :3] = rotation

    # supposing that we know an estimation of the gravity vector (e.g. along the y-axis/up vector)
    # we can try to use it to align the point clouds so that y-axis is aligned
    # here we use the y vector of the initial transformation and perturb it a bit to simulate the
    # direction of the gravity
    idx_gravity_axis = 2

    gravity_transform = gravity_transformation(
        trans_init[:3, idx_gravity_axis], gravity_axis=idx_gravity_axis
    )
    trans_init = gravity_transform @ trans_init

    trans_init = (
        align_centers_from_files(args.source, args.target, trans_init, np.eye(4))
        @ trans_init
    )
    # trans_init = np.eye(4)  # @TODO remove this to test the effect of the initial transformation

    # logger.debug(f"axis aligned:\n{trans_init @ np.eye(4)[:, idx_gravity_axis]}")
    logger.info(f"Updated initial transformation:\n{trans_init}")

    if VISUALIZE:
        draw_registration_result(
            source_raw, target_raw, trans_init, "Corrected settings", 
            size=frame_size, 
            target_frame_trans=np.eye(4), 
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

    # --------------------------------------------------------------------------------------------------
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
        T_teaser @ trans_init, source_gt_transform
    )
    matrix = T_teaser @ source_gt_transform
    logger.debug(f"Product of the transformations:\n{matrix}")
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
    )

    # Save the calculated metrics to a .json file
    output_metrics = {
        "rotation_error_rad": rot_err,
        "rotation_error_deg": np.degrees(rot_err),
        "translation_error": trans_err,
    }
    try:
        if not os.path.exists(source_dir + '/teaser_metrics'):
            os.makedirs(source_dir + '/teaser_metrics')
        pcd_file_path = args.source.replace('.ply', '_metrics.json').replace('.pcd', '_metrics.json')
        _, file_name = os.path.split(pcd_file_path)
        metrics_file = os.path.join(source_dir, 'teaser_metrics', file_name)
        with open(metrics_file, 'w') as file:
            json.dump(output_metrics, file, indent=4)
            logger.info(f"Saved metrics to {metrics_file}")
    except FileNotFoundError:
        logger.error(f"The file '{metrics_file}' was not found.")

    # --------------------------------------------------------------------------------------------------
        
    # Visualize the registration results after TEASER++
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_teaser, window_name="TEASER++ Registration Results", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4), 
                                 source_frame_trans=T_teaser @ trans_init)

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
        draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", 
                                 size=frame_size, 
                                 target_frame_trans=np.eye(4), 
                                 source_frame_trans=T_icp @ trans_init)

    ## METRICS ##
    # Calculate and save registration metrics 
    registration_metrics(target_raw, source_raw, target_down_nb_points, source_down_nb_points, teaser_solver, icp_sol, trans_init, num_corrs, NOISE_BOUND, registration_total_time, args)

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
        "--translation", type=lambda s: np.array([float(item) for item in s.split(',')]), help="Translation vector to apply to both source and target point clouds (in mm)", default="0,0,0"
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
            teaserpp_registration_real(input_args)
            count += 1
    else:
        source_dir, _ = os.path.split(input_args.source)
        teaserpp_registration_real(input_args)
