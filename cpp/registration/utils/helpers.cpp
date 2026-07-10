/// helpers.cpp – mirrors scripts/helpers.py

#include "registration/utils/helpers.hpp"
#include "registration/utils/logging.hpp"

#include <pcl/common/common.h>

#include <teaser/fpfh.h>
#include <teaser/matcher.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace registration::utils {

static auto logger = get_logger("helpers");

// ─────────────────────────────────────────────────────────────────────────────
Eigen::MatrixXd pcd2xyz(const PointCloud& pcd) {
    int n = static_cast<int>(pcd.size());
    Eigen::MatrixXd xyz(3, n);
    for (int i = 0; i < n; ++i) {
        xyz(0, i) = pcd[i].x;
        xyz(1, i) = pcd[i].y;
        xyz(2, i) = pcd[i].z;
    }
    return xyz;  // 3 × N
}

// ─────────────────────────────────────────────────────────────────────────────
teaser::FPFHCloudPtr extract_fpfh(const PointCloud& pcd, double voxel_size) {
    double radius_normal  = voxel_size * 2.0;
    double radius_feature = voxel_size * 5.0;

    teaser::PointCloud teaser_pcd = pcl_to_teaser(pcd);

    teaser::FPFHEstimation fpfh_est;
    return fpfh_est.computeFPFHFeatures(teaser_pcd, radius_normal, radius_feature);
}

// ─────────────────────────────────────────────────────────────────────────────
teaser::PointCloud pcl_to_teaser(const PointCloud& pcd) {
    teaser::PointCloud out;
    out.reserve(pcd.size());
    for (const auto& p : pcd) {
        teaser::PointXYZ tp;
        tp.x = p.x; tp.y = p.y; tp.z = p.z;
        out.push_back(tp);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
std::pair<std::vector<int>, std::vector<int>>
find_correspondences(teaser::PointCloud& src_pts,
                     teaser::PointCloud& tgt_pts,
                     teaser::FPFHCloud&  src_feats,
                     teaser::FPFHCloud&  tgt_feats,
                     bool use_crosscheck,
                     bool use_tuple_test) {
    teaser::Matcher matcher;
    auto corrs = matcher.calculateCorrespondences(
        src_pts, tgt_pts, src_feats, tgt_feats,
        /*use_absolute_scale=*/true,
        use_crosscheck,
        use_tuple_test,
        /*tuple_scale=*/0.95f);

    std::vector<int> idx0, idx1;
    idx0.reserve(corrs.size());
    idx1.reserve(corrs.size());
    for (const auto& p : corrs) {
        idx0.push_back(p.first);
        idx1.push_back(p.second);
    }
    return {idx0, idx1};
}

// ─────────────────────────────────────────────────────────────────────────────
std::pair<std::vector<int>, std::vector<int>>
find_correspondences_spatial(
    const PointCloud& source_pcd,
    const PointCloud& target_pcd,
    teaser::PointCloud& src_pts,
    teaser::PointCloud& tgt_pts,
    teaser::FPFHCloud&  src_feats,
    teaser::FPFHCloud&  tgt_feats,
    double max_distance) {

    auto [ci0, ci1] = find_correspondences(src_pts, tgt_pts, src_feats, tgt_feats);

    std::vector<int> filtered0, filtered1;
    for (int k = 0; k < static_cast<int>(ci0.size()); ++k) {
        const auto& sp = source_pcd[ci0[k]];
        const auto& tp = target_pcd[ci1[k]];
        double dx = sp.x - tp.x, dy = sp.y - tp.y, dz = sp.z - tp.z;
        if (std::sqrt(dx*dx + dy*dy + dz*dz) <= max_distance) {
            filtered0.push_back(ci0[k]);
            filtered1.push_back(ci1[k]);
        }
    }
    return {filtered0, filtered1};
}

// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<teaser::RobustRegistrationSolver>
get_teaser_solver(double noise_bound) {
    teaser::RobustRegistrationSolver::Params params;
    params.cbar2            = 0.0075;
    params.noise_bound      = noise_bound;
    params.estimate_scaling = false;
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    params.rotation_tim_graph =
        teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
    params.rotation_gnc_factor     = 1.4;
    params.rotation_max_iterations = 1000;
    params.rotation_cost_threshold = 1e-16;

    return std::make_unique<teaser::RobustRegistrationSolver>(params);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix4d Rt2T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

// ─────────────────────────────────────────────────────────────────────────────
void print_point_cloud_info(const PointCloud& pcd, const std::string& name) {
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(pcd, min_pt, max_pt);

    std::ostringstream oss;
    oss << "Point Cloud '" << name << "':\n"
        << "\tNumber of points : " << pcd.size() << "\n"
        << "\tBB extent (xyz)  : ["
            << (max_pt.x - min_pt.x) << ", "
            << (max_pt.y - min_pt.y) << ", "
            << (max_pt.z - min_pt.z) << "]";
    logger.debug(oss.str());
}

} // namespace registration::utils
