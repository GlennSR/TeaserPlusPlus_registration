#pragma once
/// metrics.hpp – mirrors registration/utils/metrics.py

#include "registration/utils/helpers.hpp"  // PointCloud alias

#include <Eigen/Dense>
#include <teaser/registration.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace registration::utils {

// ─── Local substitute for Open3D RegistrationResult ─────────────────────────
struct RegistrationResult {
    double fitness_       = 0.0;
    double inlier_rmse_   = 0.0;
    /// Pairs of (source_idx, target_idx) for points within the distance threshold
    std::vector<std::pair<int, int>> correspondence_set_;
    Eigen::Matrix4d transformation_ = Eigen::Matrix4d::Identity();
};

/// Evaluate registration quality using a KDTree distance-threshold approach.
/// Equivalent to open3d::pipelines::registration::EvaluateRegistration().
/// @param max_distance  Correspondence distance threshold (same units as cloud)
RegistrationResult evaluate_registration(
    const PointCloud& source,
    const PointCloud& target,
    double max_distance,
    const Eigen::Matrix4d& transformation = Eigen::Matrix4d::Identity());

/// Nearest-neighbour distance from every source point to the closest target point.
/// Equivalent to open3d PointCloud::ComputePointCloudDistance().
std::vector<double> compute_point_cloud_distance(
    const PointCloud& source,
    const PointCloud& target);

/// ──────────────────────────────────────────────────────────────────────────
/// Low-level distance metrics

/// RMSE between corresponding points at the same index in two clouds.
std::tuple<double, std::vector<double>>
compute_rmse_between_point_clouds(
    const PointCloud& source,
    const PointCloud& target);

/// RMSE between two transformations applied to the same point cloud.
double compute_rmse_transformations(
    const Eigen::Matrix4d& transf_est,
    const Eigen::Matrix4d& transf_gt,
    const PointCloud& pcd);

/// ──────────────────────────────────────────────────────────────────────────
/// Registration evaluation bundle

struct RegistrationArgs {
    std::string source;
    std::string target;
    double      voxel_size            = 30.0;
    double      refinement_voxel_size = 0.0;
};

/// Compute and log all registration quality metrics, then save a JSON file.
void registration_metrics(
    const PointCloud& target_raw,
    const PointCloud& source_raw,
    int target_down_nb_points,
    int source_down_nb_points,
    teaser::RobustRegistrationSolver& teaser_solver,
    const RegistrationResult& icp_sol,
    const Eigen::Matrix4d& trans_init,
    int num_corrs,
    double noise_bound,
    double registration_total_time,
    const RegistrationArgs& args,
    const std::string& output_dir);

/// Load a ground-truth 4×4 transform from a JSON file.
/// Translation is converted from metres to millimetres.
/// Returns identity if the file is not found.
Eigen::Matrix4d load_gt_transform(const std::string& json_file);

/// Calculate TEASER++ errors against ground truth and save a metrics JSON.
void calculate_errors(
    const RegistrationArgs& args,
    const RegistrationResult* icp_sol,
    const Eigen::Matrix4d& estimated_transform,
    double voxel_size,
    const std::string& scan_gt_json,
    double total_time,
    const PointCloud& source,
    const PointCloud& target,
    const std::string& output_dir = "metrics/");

/// Save an estimated 4×4 transformation to a JSON file (key "H").
void save_estimated_poses(
    const Eigen::Matrix4d& estimated_transform,
    const std::string& source_path,
    const std::string& output_dir);

} // namespace registration::utils
