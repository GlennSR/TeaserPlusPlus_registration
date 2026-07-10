#pragma once
/// point_cloud.hpp – mirrors registration/utils/point_cloud.py

#include <Eigen/Dense>
#include <string>
#include <tuple>

#include <teaser/fpfh.h>

#include "registration/utils/helpers.hpp"  // PointCloud alias + extract_fpfh

namespace registration::utils {

/// Downsample and compute FPFH features.
/// Returns (downsampled_pcd, FPFHCloudPtr).
std::tuple<PointCloud, teaser::FPFHCloudPtr>
preprocess_point_cloud(const PointCloud& pcd, double voxel_size);

/// Load a PLY file, optionally downsample and estimate normals.
/// Scales the cloud to mm (×1000).
PointCloud load_point_cloud(const std::string& ply_path,
                             double voxel_size     = 0.0,
                             bool estimate_normals = true);

/// Overload: accepts an existing PointCloud (deep copy).
PointCloud load_point_cloud(const PointCloud& pcd,
                             double voxel_size     = 0.0,
                             bool estimate_normals = true);

/// Load source + target for ICP refinement from file paths.
/// Applies trans_init to the source. Returns (source_down, target_down).
std::tuple<PointCloud, PointCloud>
load_point_clouds_files_for_refinement(
    const std::string& source_ply,
    const std::string& target_ply,
    double             voxel_size,
    const Eigen::Matrix4d& trans_init = Eigen::Matrix4d::Identity());

/// Same as above but accepts already-loaded PointCloud objects.
std::tuple<PointCloud, PointCloud>
load_point_clouds_for_refinement(
    const PointCloud& source,
    const PointCloud& target,
    double            voxel_size,
    const Eigen::Matrix4d& trans_init = Eigen::Matrix4d::Identity());

/// Add Gaussian noise (std = std_dev) to every point in the cloud.
void noise_gaussian(PointCloud& pcd, double std_dev);

/// Estimate a "rough scale" of the point cloud (nearest power of 10 to the
/// largest extent of the axis-aligned bounding box).
double rough_scale_point_cloud(const PointCloud& pcd);

/// Same as above, loading from a file.
double rough_scale_point_cloud_from_file(const std::string& ply_path);

/// Compute a pure-translation 4×4 matrix that aligns source centroid to target.
Eigen::Matrix4d align_centers(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& trans_init  = Eigen::Matrix4d::Identity(),
    const Eigen::Matrix4d& correction  = Eigen::Matrix4d::Identity());

/// Same as align_centers but loads from files.
Eigen::Matrix4d align_centers_from_files(
    const std::string& source_file,
    const std::string& target_file,
    const Eigen::Matrix4d& trans_init  = Eigen::Matrix4d::Identity(),
    const Eigen::Matrix4d& correction  = Eigen::Matrix4d::Identity());

/// Remove points farther than max_distance from the centroid.
PointCloud filter_points_far_from_center(const PointCloud& pcd,
                                          double max_distance = 10.0);

/// Apply a 4×4 rigid transform to every point in the cloud (in-place).
void transform_point_cloud(PointCloud& pcd, const Eigen::Matrix4d& T);

/// Return a transformed copy of the cloud.
PointCloud transform_point_cloud_copy(const PointCloud& pcd,
                                       const Eigen::Matrix4d& T);

} // namespace registration::utils

