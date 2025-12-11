import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from registration.utils.point_cloud import rough_scale_point_cloud, align_centers_from_files, align_centers
from registration.utils.transforms import (generate_random_rotation_matrix, 
    perturb_direction, 
    rotation_aligning_two_directions, 
    rototranslation_from_rotation_translation)


def gravity_transformation(
    gravity_direction: np.ndarray, gravity_axis: int = 1
) -> np.ndarray:
    """Compute a transformation matrix to align a given gravity direction.

    This function computes a rotation matrix that aligns the specified gravity
    direction with the desired gravity axis (default is y-axis). This can be useful if the
    gravity vector is given by an IMU sensor in a reference system similar (same up direction)
    to the one of the point cloud.
    It returns a 4x4 transformation matrix that can be applied to point clouds so that the
    point cloud is aligned with the gravity direction.

    Args:
        gravity_direction: A 3D vector representing the measured gravity direction in the point cloud's reference frame.
        gravity_axis: The axis index (0 for x, 1 for y, 2 for z) to align the gravity direction to.

    Returns:
        A 4x4 transformation matrix with null translation that aligns the gravity direction with the specified axis.
    """
    if gravity_axis < 0 or gravity_axis > 2:
        raise ValueError("gravity_axis must be 0 (x), 1 (y), or 2 (z)")

    # @TODO this is to add some noise to the gravity direction
    dst_gravity_direction = perturb_direction(
        np.eye(3)[:, gravity_axis], sigma=np.deg2rad(1)
    )
    dst_gravity_direction = np.eye(3)[:, gravity_axis]

    gravity_aligned_rotation = rotation_aligning_two_directions(
        gravity_direction, dst_gravity_direction
    )
    gravity_transform = rototranslation_from_rotation_translation(
        gravity_aligned_rotation, np.zeros(3)
    )
    return gravity_transform

def prepare_dataset(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> tuple:
    """Load and prepare point cloud datasets for registration.

    Loads source and target point clouds from files, applies an initial transformation
    to the source cloud, and preprocesses both clouds by downsampling and computing
    FPFH features for feature-based registration.

    Args:
        source_file: File path to the source point cloud.
        target_file: File path to the target point cloud.
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
    #logger.info("Load two point clouds and disturb initial pose")
    
    source.transform(correction)
    
    target.transform(correction)

    source.transform(trans_init)

    transf = align_centers(source, target, np.eye(4), np.eye(4))
    trans_init = transf @ trans_init
    
    source.transform(transf)

    return source, target

VOXEL_SIZE = 30
VISUALIZE = True

# Load and visualize two point clouds from 3DMatch dataset
A_pcd_raw = o3d.io.read_point_cloud('/home/gro5293/pcl_registration/teaserpp/data/sameref/E_shape_maq15k.ply')
B_pcd_raw = o3d.io.read_point_cloud('/home/gro5293/pcl_registration/teaserpp/data/sameref/maquette300k.ply')
A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw], window_name="Initial State (Source: Blue, Target: Red)") # plot A and B 

# voxel downsample both clouds
A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd], window_name="Downsampled Point Clouds") # plot downsampled A and B 

frame_size = 2000
trans_init = np.asarray(
        [
            [0.862, 0.011, -0.507, 3.10005 * frame_size],
            [-0.139, 0.967, -0.215, 3.51007 * frame_size],
            [0.487, 0.255, 0.835, -0.4 * frame_size],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

trans_init[:3, :3] = generate_random_rotation_matrix()

# AFTER THIS WAS ADDED THE PCLs INITIAL STATE BECAME WORSE
# ------------------------------------------------------------------------
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
    align_centers(A_pcd, B_pcd, trans_init, np.eye(4))
    @ trans_init
)
# ------------------------------------------------------------------------

if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd], window_name="Random transform") # plot downsampled A and B 

A_pcd, B_pcd = prepare_dataset(A_pcd, B_pcd)

if VISUALIZE:
    o3d.visualization.draw_geometries([A_pcd,B_pcd], window_name="Aligned centers") # plot downsampled A and B 

A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

# extract FPFH features
A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)

# establish correspondences by nearest neighbour search in feature space
corrs_A, corrs_B = find_correspondences(
    A_feats, B_feats, mutual_filter=True)
A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

num_corrs = A_corr.shape[1]
print(f'FPFH generates {num_corrs} putative correspondences.')

# visualize the point clouds together with feature correspondences
points = np.concatenate((A_corr.T,B_corr.T),axis=0)
lines = []
for i in range(num_corrs):
    lines.append([i,i+num_corrs])
colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set], window_name="FPFH Correspondences")

# robust global registration using TEASER++
NOISE_BOUND = VOXEL_SIZE
teaser_solver = get_teaser_solver(NOISE_BOUND)
teaser_solver.solve(A_corr,B_corr)
solution = teaser_solver.getSolution()
R_teaser = solution.rotation
t_teaser = solution.translation
T_teaser = Rt2T(R_teaser,t_teaser)

# Visualize the registration results
A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd], window_name="TEASER++ Registration Results")

# local refinement using ICP
# icp_sol = o3d.pipelines.registration.registration_icp(
#       A_pcd, B_pcd, NOISE_BOUND, T_teaser,
#       o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#       o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))

icp_sol = o3d.pipelines.registration.registration_icp(
      A_pcd, B_pcd, NOISE_BOUND, T_teaser,
      o3d.pipelines.registration.TransformationEstimationPointToPlane(),)

T_icp = icp_sol.transformation

# visualize the registration after ICP refinement
A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd], window_name="ICP Refinement")


