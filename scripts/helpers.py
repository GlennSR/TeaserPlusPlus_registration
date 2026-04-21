import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree
import teaserpp_python

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def find_correspondences_spatial(source_pcd, target_pcd, feats0, feats1, max_distance, mutual_filter=True):
  """Find feature correspondences with an additional spatial distance filter.

  First finds correspondences using nearest-neighbour search in FPFH feature space
  (with optional mutual/cross-check filter), then removes pairs whose Euclidean
  distance in 3D exceeds max_distance.

  Args:
      source_pcd: Source point cloud (open3d.geometry.PointCloud).
      target_pcd: Target point cloud (open3d.geometry.PointCloud).
      feats0: FPFH features of the source point cloud (N x 33 array).
      feats1: FPFH features of the target point cloud (M x 33 array).
      max_distance: Maximum allowed Euclidean distance between matched points.
      mutual_filter: If True, only keep mutually consistent matches.

  Returns:
      Tuple of (corres_idx0, corres_idx1) — arrays of corresponding indices
      in source and target that pass both feature and spatial filtering.
  """
  # Step 1: Feature-based correspondences
  corres_idx0, corres_idx1 = find_correspondences(feats0, feats1, mutual_filter=mutual_filter)

  # Step 2: Spatial distance filter
  source_pts = np.asarray(source_pcd.points)
  target_pts = np.asarray(target_pcd.points)

  dists = np.linalg.norm(source_pts[corres_idx0] - target_pts[corres_idx1], axis=1)
  spatial_mask = dists <= max_distance

  return corres_idx0[spatial_mask], corres_idx1[spatial_mask]

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 0.0075 # truncation_distance² = cbar2 * noise_bound²
    # cbar2 = 1.25 for simulated dataset
    # cbar2 = 0.25 for real dataset
    # cbar2 < 1 more agressive outlier rejection, cbar2 > 1 more tolerant, standard is 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.QUATRO
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 1000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def get_teaser_solver_test(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 0.0075 # truncation_distance² = cbar2 * noise_bound²
    print(f"Using cbar2 = {solver_params.cbar2} (truncation_distance² = cbar2 * noise_bound²)")
    # cbar2 < 1 more agressive outlier rejection, cbar2 > 1 more tolerant, standard is 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.QUATRO
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 1000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 