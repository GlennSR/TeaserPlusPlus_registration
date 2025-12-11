import argparse
import logging
import os
from registration.utils.logging import setup_logging

import open3d as o3d
from registration.visualization.viewer import *
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from registration.utils.point_cloud import preprocess_point_cloud, align_centers_from_files, align_centers, rough_scale_point_cloud_from_file
from registration.utils.transforms import (generate_random_rotation_matrix,
    transformation_error,
    gravity_transformation)

logger = logging.getLogger(__name__)

def prepare_dataset(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> tuple:
    """Load and prepare point cloud datasets for registration.
    Eu estive aqui. 
    EU SEI
    Loads source and target downsampled point clouds, applies an initial transformation
    to the source cloud, and preprocesses both clouds by computing
    FPFH features for feature-based registration.

    Args:
        source_down: Downsampled source point cloud.
        target_down: Downsampled target point cloud.
        voxel_size: The size of the voxel for downsampling both point clouds.
        trans_init: Initial transformation matrix to apply to the source cloud (default: identity matrix).
        correction: Correction transformation matrix to apply to both clouds, typically to align to the visual reference frame (default: identity matrix).

    Returns:
        A tuple containing:
            - source_down: Downsampled source point cloud
            - target_down: Downsampled target point cloud
            - source_fpfh: FPFH features of the downsampled source
            - target_fpfh: FPFH features of the downsampled target
    """
    #logger.info("Load two point clouds and disturb initial pose")
    
    source.transform(correction)
    
    target.transform(correction)

    source.transform(trans_init)

    # transf = align_centers(source_down, target_down, np.eye(4), np.eye(4))
    # trans_init = transf @ trans_init
    # print(f"transf: {transf} \ntrans_init: {trans_init}")
    
    # source_down.transform(transf)

    logger.info("Preprocessing source point cloud")
    source_down, source_fpfh = preprocess_point_cloud(logger, source, voxel_size)
    print_point_cloud_info(source_down, "Downsampled source")
    logger.info(f"Feature of SOURCE: {source_fpfh}")

    logger.info("Preprocessing target point cloud")
    target_down, target_fpfh = preprocess_point_cloud(logger, target, voxel_size)
    print_point_cloud_info(target_down, "Downsampled target")
    logger.info(f"Feature of TARGET: {target_fpfh}")

    return source, target, source_down, target_down, source_fpfh, target_fpfh

def is_solution_upside_down(transformation: np.ndarray, idx_gravity_axis: int) -> bool:
    """Check if the given transformation results in an upside-down alignment.

    This function examines the rotation component of the provided transformation
    matrix to determine if the direction corresponding to the specified gravity axis
    is inverted (i.e., points in the opposite direction). This can be useful for
    validating registration results against expected orientations.

    Args:
        transformation: A 4x4 transformation matrix to evaluate.
        idx_gravity_axis: The index of the gravity axis (0 for x, 1 for y, 2 for z).
    Returns:
        True if the solution is upside down, False otherwise.
    Raises:
        ValueError: If idx_gravity_axis is not 0, 1, or 2.
    """
    if idx_gravity_axis < 0 or idx_gravity_axis > 2:
        raise ValueError("idx_gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    gravity = np.eye(3)[:, idx_gravity_axis]
    direction = transformation[:3, :3] @ gravity
    return np.dot(direction, gravity) < 0

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
    VOXEL_SIZE = args.voxel_size
    VISUALIZE = args.viz

    source_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show source in blue
    target_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show target in red
    frame_size = rough_scale_point_cloud_from_file(args.target)

    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, np.eye(4), window_name="Initial State (Source: Blue, Target: Red)", size=frame_size)

    trans_init = np.asarray(
            [
                [0.862, 0.011, -0.507, 3.10005 * frame_size],
                [-0.139, 0.967, -0.215, 3.51007 * frame_size],
                [0.487, 0.255, 0.835, -0.4 * frame_size],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    trans_init[:3, :3] = generate_random_rotation_matrix()
    trans_init = np.eye(4)

    # supposing that we know an estimation of the gravity vector (e.g. along the y-axis/up vector)
    # we can try to use it to align the point clouds so that y-axis is aligned
    # here we use the y vector of the initial transformation and perturb it a bit to simulate the
    # direction of the gravity
    idx_gravity_axis = 1

    gravity_transform = gravity_transformation(
        trans_init[:3, idx_gravity_axis], gravity_axis=idx_gravity_axis
    )
    trans_init = gravity_transform @ trans_init

    trans_init = (
        align_centers_from_files(args.source, args.target, trans_init, np.eye(4))
        @ trans_init
    )
    
    logger.debug(f"axis aligned:\n{trans_init @ np.eye(4)[:, idx_gravity_axis]}")

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

    # ---- VISUALIZATION PURPOSE ONLY ----
    # visualize the point clouds together with feature correspondences
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
    if VISUALIZE:
        o3d.visualization.draw_geometries([source_raw, target_raw, line_set], window_name="FPFH Correspondences")
    # ---- VISUALIZATION PURPOSE ONLY ----

    NOISE_BOUND = VOXEL_SIZE * 2 # 2 works well
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(source_corr,target_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)
    
    # Log TEASER++ internal quality metrics
    translation_inliers = teaser_solver.getTranslationInliers()
    rotation_inliers = teaser_solver.getRotationInliers()
    logger.info(f"TEASER++ Internal Metrics:")
    logger.info(f"  Translation inliers: {translation_inliers} / {num_corrs} ({len(translation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Rotation inliers (max clique): {len(rotation_inliers)} / {num_corrs} ({len(rotation_inliers)/num_corrs*100:.1f}%)")
    logger.info(f"  Solution valid: {solution.valid if hasattr(solution, 'valid') else 'N/A'}")

    # # Check if upside down
    # if is_solution_upside_down(T_teaser, idx_gravity_axis):
    #     logger.warning(f"TEASER++ attempt result is upside down, discarding.")
    

    # Evaluate the solution using Open3D
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_raw, target_raw, NOISE_BOUND, T_teaser
    )
    logger.info(f"Open3D Evaluation Metrics:")
    logger.info(f"  Fitness: {evaluation.fitness:.4f} (fraction of inlier points)")
    logger.info(f"  Inlier RMSE: {evaluation.inlier_rmse:.6f} (lower is better)")
    logger.info(f"  Correspondence set size: {len(evaluation.correspondence_set)}")
        
    # Visualize the registration results
    source_raw_T_teaser = copy.deepcopy(source_raw).transform(T_teaser)
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_teaser, window_name="TEASER++ Registration Results", size=frame_size)

    # local refinement using ICP Point to Plane
    icp_sol = refine_registration(source_raw, target_raw, NOISE_BOUND, T_teaser, max_iteration=args.max_iter_icp)
    T_icp = icp_sol.transformation

    logger.info(f"ICP refinement result: {icp_sol}")
    logger.info(f"Estimated matrix:\n{icp_sol.transformation}")
    logger.info(
        f"Result fitness: {icp_sol.fitness}, inlier RMSE: {icp_sol.inlier_rmse}"
    )

    # NB this only make sense if you are aligning the same model
    # difference between initial and final transformation
    frame_size = 1
    trans_init = np.asarray(
            [
                [1, 0, 0, 0 * frame_size],
                [0, 1, 0, -1 * frame_size],
                [0, 0, 1, 1 * frame_size],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    rot_err, trans_err = transformation_error(
        icp_sol.transformation, np.linalg.inv(trans_init) # TODO: change to .json matrix
    )
    logger.info(
        f"Rotation error (radians): {rot_err:.4f} (degrees: {np.degrees(rot_err):.4f}), Translation error: {trans_err:.4f}"
    )

    # visualize the registration after ICP refinement
    source_raw_T_icp = copy.deepcopy(source_raw).transform(T_icp)
    if VISUALIZE:
        draw_registration_result(source_raw, target_raw, T_icp, window_name="ICP Refinement", size=frame_size)

    # Calculate the metric of the result transformation using Open3D compute_point_cloud_distance() method
    distances_o3d = target_raw.compute_point_cloud_distance(source_raw_T_icp)
    # logger.info(f"Distances for the registration result: {distances_o3d}")
    logger.info(f"Mean Open3D distance for the registration result: {np.mean(distances_o3d):.6f}")


if __name__ == "__main__":
    # tutorial from here https://teaser.readthedocs.io/en/master/quickstart.html
    
    # add input file argument
    parser = argparse.ArgumentParser(description="Teaser++ registration")
    parser.add_argument("--source", type=str, help="source file path", required=True)
    parser.add_argument("--target", type=str, help="taraget file path", required=True)

    parser.add_argument(
        "--voxel-size", type=float, help="voxels size for downsampling", default=0.05
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
        help="Set logging level (default: WARNING)",
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
        teaserpp_registration(input_args)

