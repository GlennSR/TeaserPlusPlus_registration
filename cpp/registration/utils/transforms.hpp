#pragma once
/// Transformation and rotation utilities – mirrors transforms.py

#include <Eigen/Dense>
#include <tuple>

namespace registration::utils {

/// Combine a 3x3 rotation and 3D translation into a 4x4 homogeneous matrix.
Eigen::Matrix4d rototranslation_from_rotation_translation(
    const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans);

/// Convert a rotation matrix to (axis, angle) representation.
/// Returns (unit_axis, angle_radians).
std::tuple<Eigen::Vector3d, double> axis_angle_from_rotation(
    const Eigen::Matrix3d& rot_mat);

/// 3x3 skew-symmetric cross-product matrix for vector v.
Eigen::Matrix3d cross_matrix(const Eigen::Vector3d& v);

/// Rotation matrix from axis-angle (Rodrigues' formula).
Eigen::Matrix3d rotation_matrix_from_axis_angle(
    const Eigen::Vector3d& axis, double angle);

/// Geodesic angular error between two rotation matrices [radians, 0..π].
double rotation_error_angle(
    const Eigen::Matrix3d& rot_est, const Eigen::Matrix3d& rot_gt);

/// Translation error accounting for rotation difference.
/// Returns (norm, error_vector).
std::tuple<double, Eigen::Vector3d> translation_error(
    const Eigen::Matrix3d& rot_est, const Eigen::Vector3d& t_est,
    const Eigen::Matrix3d& rot_gt, const Eigen::Vector3d& t_gt);

/// Both rotation and translation errors from 4x4 matrices.
/// Returns (rot_err_radians, trans_err_norm).
std::tuple<double, double> transformation_error(
    const Eigen::Matrix4d& t_est, const Eigen::Matrix4d& t_gt);

/// Uniformly random rotation matrix sampled from SO(3) via QR decomposition.
Eigen::Matrix3d generate_random_rotation_matrix();

/// True if mat is a valid rotation matrix (orthogonal, det = +1).
bool is_rotation_matrix(const Eigen::Matrix3d& mat);

/// Rotation matrix that aligns src_dir onto tgt_dir (R @ src = tgt).
Eigen::Matrix3d rotation_aligning_two_directions(
    const Eigen::Vector3d& src_dir, const Eigen::Vector3d& tgt_dir);

/// Perturb a unit direction by a small random rotation (std = sigma radians).
Eigen::Vector3d perturb_direction(const Eigen::Vector3d& direction, double sigma);

/// Small random rotation matrix (axis sampled uniformly, angle ~ N(0, sigma)).
Eigen::Matrix3d random_small_rotation(double sigma);

/// Perturb a rotation matrix by a small random rotation.
Eigen::Matrix3d perturb_rotation_matrix(const Eigen::Matrix3d& rot_mat, double sigma);

/// Elementary rotation matrices.
Eigen::Matrix3d rot_mat_x(double angle);
Eigen::Matrix3d rot_mat_y(double angle);
Eigen::Matrix3d rot_mat_z(double angle);

/// 4x4 flip transformation (±90° rotation around the specified axis).
/// Valid axis values: "x", "nx", "y", "ny", "z", "nz".
Eigen::Matrix4d get_flip_transform(const std::string& axis);

} // namespace registration::utils
