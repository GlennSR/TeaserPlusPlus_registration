#pragma once
/// viewer.hpp – mirrors registration/visualization/viewer.py

#include "registration/utils/helpers.hpp"  // PointCloud alias

#include <Eigen/Dense>
#include <string>

namespace registration::visualization {

/// Visualise the registration result in an interactive PCLVisualizer window.
/// Source is rendered yellow, target cyan.
/// Coordinate frames are drawn at world origin, target frame, and source centre.
void draw_registration_result(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& transformation,
    const std::string& window_name,
    double size                                = 1.0,
    const Eigen::Matrix4d& target_frame_trans  = Eigen::Matrix4d::Identity(),
    const Eigen::Matrix4d& source_frame_trans  = Eigen::Matrix4d::Identity());

/// Same as draw_registration_result – kept for API compatibility.
void save_registration_result(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& transformation,
    const std::string& window_name,
    double size = 1.0);

/// Print basic point-cloud statistics.
void print_point_cloud_info(
    const PointCloud& pcd,
    const std::string& name = "Point cloud");

} // namespace registration::visualization
