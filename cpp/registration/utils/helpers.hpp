#pragma once
/// helpers.hpp – mirrors scripts/helpers.py
/// Provides: pcd2xyz, extract_fpfh, find_correspondences,
///           find_correspondences_spatial, get_teaser_solver, Rt2T.

#include <Eigen/Dense>
#include <memory>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>

#include <teaser/fpfh.h>
#include <teaser/matcher.h>
#include <teaser/registration.h>

// ─── Shared point-cloud type aliases (global scope so all modules can use) ────
using PointCloud    = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;

namespace registration::utils {

/// Convert a PCL point cloud to a 3×N Eigen matrix (one column per point).
Eigen::MatrixXd pcd2xyz(const PointCloud& pcd);

/// Compute FPFH features using teaser::FPFHEstimation.
/// Returns a teaser::FPFHCloudPtr (pcl::PointCloud<pcl::FPFHSignature33>::Ptr).
teaser::FPFHCloudPtr extract_fpfh(const PointCloud& pcd, double voxel_size);

/// Feature-space correspondence search using teaser::Matcher
/// (mutual cross-check + optional tuple test).
/// Returns (corres_idx_source, corres_idx_target).
std::pair<std::vector<int>, std::vector<int>>
find_correspondences(teaser::PointCloud& src_pts,
                     teaser::PointCloud& tgt_pts,
                     teaser::FPFHCloud&  src_feats,
                     teaser::FPFHCloud&  tgt_feats,
                     bool use_crosscheck  = true,
                     bool use_tuple_test  = false);

/// Same as find_correspondences, but also rejects pairs whose 3-D Euclidean
/// distance exceeds max_distance.
std::pair<std::vector<int>, std::vector<int>>
find_correspondences_spatial(
    const PointCloud& source_pcd,
    const PointCloud& target_pcd,
    teaser::PointCloud& src_pts,
    teaser::PointCloud& tgt_pts,
    teaser::FPFHCloud&  src_feats,
    teaser::FPFHCloud&  tgt_feats,
    double max_distance);

/// Convert a PCL PointCloud to teaser::PointCloud.
teaser::PointCloud pcl_to_teaser(const PointCloud& pcd);

/// Build and return a configured TEASER++ solver (mirrors get_teaser_solver()).
std::unique_ptr<teaser::RobustRegistrationSolver>
get_teaser_solver(double noise_bound);

/// Assemble a 4×4 homogeneous transformation from R (3×3) and t (3×1).
Eigen::Matrix4d Rt2T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

/// Log basic statistics of a point cloud (number of points, bounding box).
void print_point_cloud_info(const PointCloud& pcd,
                             const std::string& name = "Point cloud");

} // namespace registration::utils
