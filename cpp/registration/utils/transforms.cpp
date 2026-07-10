/// transforms.cpp – mirrors transforms.py

#include "registration/utils/transforms.hpp"

#include <Eigen/QR>
#include <cmath>
#include <random>
#include <stdexcept>

namespace registration::utils {

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix4d rototranslation_from_rotation_translation(
    const Eigen::Matrix3d& rot, const Eigen::Vector3d& trans) {
    if (rot.rows() != 3 || rot.cols() != 3)
        throw std::invalid_argument("Rotation matrix must be 3x3.");
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = rot;
    T.block<3, 1>(0, 3) = trans;
    return T;
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<Eigen::Vector3d, double> axis_angle_from_rotation(
    const Eigen::Matrix3d& rot_mat) {
    constexpr double eps = 1e-12;
    double trace_val = (rot_mat.trace() - 1.0) / 2.0;
    double angle = std::acos(std::clamp(trace_val, -1.0, 1.0));

    if (std::abs(angle) < 1e-8) {
        return {Eigen::Vector3d(1.0, 0.0, 0.0), 0.0};
    }

    if (std::abs(angle - M_PI) < 1e-6) {
        // 180° – extract from diagonal
        Eigen::Vector3d diag = rot_mat.diagonal();
        Eigen::Vector3d axis = ((diag.array() + 1.0).max(0.0).sqrt() / std::sqrt(2.0)).matrix();
        axis /= (axis.norm() + eps);
        return {axis, angle};
    }

    Eigen::Vector3d axis(
        rot_mat(2, 1) - rot_mat(1, 2),
        rot_mat(0, 2) - rot_mat(2, 0),
        rot_mat(1, 0) - rot_mat(0, 1));
    axis /= (2.0 * std::sin(angle));
    axis /= (axis.norm() + eps);
    return {axis, angle};
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d cross_matrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<  0.0,  -v(2),  v(1),
          v(2),   0.0, -v(0),
         -v(1),   v(0),  0.0;
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d rotation_matrix_from_axis_angle(
    const Eigen::Vector3d& axis, double angle) {
    Eigen::Vector3d k = axis.normalized();
    Eigen::Matrix3d K = cross_matrix(k);
    return std::cos(angle) * Eigen::Matrix3d::Identity()
         + std::sin(angle) * K
         + (1.0 - std::cos(angle)) * (k * k.transpose());
}

// ─────────────────────────────────────────────────────────────────────────────
double rotation_error_angle(
    const Eigen::Matrix3d& rot_est, const Eigen::Matrix3d& rot_gt) {
    Eigen::Matrix3d rot_err = rot_est * rot_gt.transpose();
    double trace_val = std::clamp((rot_err.trace() - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(trace_val);
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<double, Eigen::Vector3d> translation_error(
    const Eigen::Matrix3d& rot_est, const Eigen::Vector3d& t_est,
    const Eigen::Matrix3d& rot_gt, const Eigen::Vector3d& t_gt) {
    Eigen::Matrix3d rot_err = rot_est * rot_gt.transpose();
    Eigen::Vector3d t_err = t_est - rot_err * t_gt;
    return {t_err.norm(), t_err};
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<double, double> transformation_error(
    const Eigen::Matrix4d& t_est, const Eigen::Matrix4d& t_gt) {
    if (t_est.rows() != 4 || t_gt.rows() != 4)
        throw std::invalid_argument("Both T_est and T_gt must be 4x4 matrices.");

    Eigen::Matrix3d rot_est = t_est.block<3, 3>(0, 0);
    Eigen::Vector3d tra_est = t_est.block<3, 1>(0, 3);
    Eigen::Matrix3d rot_gt  = t_gt.block<3, 3>(0, 0);
    Eigen::Vector3d tra_gt  = t_gt.block<3, 1>(0, 3);

    double rot_err = rotation_error_angle(rot_est, rot_gt);
    auto [trans_err, _] = translation_error(rot_est, tra_est, rot_gt, tra_gt);
    return {rot_err, trans_err};
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d generate_random_rotation_matrix() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::Matrix3d A;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A(i, j) = dist(rng);

    // QR decomposition → Q is orthogonal
    Eigen::HouseholderQR<Eigen::Matrix3d> qr(A);
    Eigen::Matrix3d Q = qr.householderQ() * Eigen::Matrix3d::Identity();

    // Ensure det(Q) = +1 (proper rotation)
    if (Q.determinant() < 0.0)
        Q.col(2) *= -1.0;

    return Q;
}

// ─────────────────────────────────────────────────────────────────────────────
bool is_rotation_matrix(const Eigen::Matrix3d& mat) {
    if (!mat.isApprox(mat.transpose() * mat * Eigen::Matrix3d::Identity(), 1e-6))
        return false;  // not orthogonal
    if (std::abs(mat.determinant() - 1.0) > 1e-6)
        return false;
    // re-check: R^T R ≈ I
    return (mat.transpose() * mat).isApprox(Eigen::Matrix3d::Identity(), 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d rotation_aligning_two_directions(
    const Eigen::Vector3d& src_dir, const Eigen::Vector3d& tgt_dir) {
    Eigen::Vector3d src = src_dir.normalized();
    Eigen::Vector3d tgt = tgt_dir.normalized();
    double dot = src.dot(tgt);

    if (std::abs(dot - 1.0) < 1e-10)
        return Eigen::Matrix3d::Identity();

    if (std::abs(dot + 1.0) < 1e-10) {
        // Opposite vectors – 180° rotation around any orthogonal axis
        Eigen::Vector3d ortho;
        if (std::abs(src(0)) > 1e-6 || std::abs(src(1)) > 1e-6)
            ortho = Eigen::Vector3d(-src(1), src(0), 0.0).normalized();
        else
            ortho = Eigen::Vector3d(0.0, -src(2), src(1)).normalized();
        return rotation_matrix_from_axis_angle(ortho, M_PI);
    }

    Eigen::Vector3d v = src.cross(tgt).normalized();
    double angle = std::acos(std::clamp(dot, -1.0, 1.0));
    return rotation_matrix_from_axis_angle(v, angle);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Vector3d perturb_direction(const Eigen::Vector3d& direction, double sigma) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::Vector3d random_axis(dist(rng), dist(rng), dist(rng));
    random_axis -= random_axis.dot(direction) * direction;
    random_axis.normalize();

    std::normal_distribution<double> angle_dist(0.0, sigma);
    double angle = angle_dist(rng);
    Eigen::Vector3d new_dir = rotation_matrix_from_axis_angle(random_axis, angle) * direction;
    return new_dir.normalized();
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d random_small_rotation(double sigma) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, sigma);

    Eigen::Vector3d axis(dist(rng), dist(rng), dist(rng));
    double theta = axis.norm();
    if (theta < 1e-12) return Eigen::Matrix3d::Identity();

    Eigen::Vector3d k = axis / theta;
    Eigen::Matrix3d K = cross_matrix(k);
    return Eigen::Matrix3d::Identity()
         + std::sin(theta) * K
         + (1.0 - std::cos(theta)) * (K * K);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d perturb_rotation_matrix(
    const Eigen::Matrix3d& rot_mat, double sigma) {
    if (!is_rotation_matrix(rot_mat))
        throw std::invalid_argument("Input matrix must be a valid rotation matrix.");
    return random_small_rotation(sigma) * rot_mat;
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix3d rot_mat_x(double angle) {
    return rotation_matrix_from_axis_angle(Eigen::Vector3d::UnitX(), angle);
}
Eigen::Matrix3d rot_mat_y(double angle) {
    return rotation_matrix_from_axis_angle(Eigen::Vector3d::UnitY(), angle);
}
Eigen::Matrix3d rot_mat_z(double angle) {
    return rotation_matrix_from_axis_angle(Eigen::Vector3d::UnitZ(), angle);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix4d get_flip_transform(const std::string& axis) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    auto make = [&](Eigen::Matrix3d R) { T.block<3, 3>(0, 0) = R; };

    if      (axis == "x")  make(rot_mat_x( M_PI / 2.0));
    else if (axis == "nx") make(rot_mat_x(-M_PI / 2.0));
    else if (axis == "y")  make(rot_mat_y( M_PI / 2.0));
    else if (axis == "ny") make(rot_mat_y(-M_PI / 2.0));
    else if (axis == "z")  make(rot_mat_z( M_PI / 2.0));
    else if (axis == "nz") make(rot_mat_z(-M_PI / 2.0));
    else throw std::invalid_argument("Invalid flip axis: " + axis);

    return T;
}

} // namespace registration::utils
