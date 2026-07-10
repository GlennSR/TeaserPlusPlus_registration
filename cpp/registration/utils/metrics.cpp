/// metrics.cpp – mirrors registration/utils/metrics.py

#include "registration/utils/metrics.hpp"
#include "registration/utils/logging.hpp"
#include "registration/utils/transforms.hpp"
#include "registration/utils/point_cloud.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace registration::utils {

static auto logger = get_logger("metrics");

// ─────────────────────────────────────────────────────────────────────────────
// PCL-based equivalents for Open3D evaluation utilities
// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> compute_point_cloud_distance(
    const PointCloud& source,
    const PointCloud& target) {

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target.makeShared());

    std::vector<double> dists;
    dists.reserve(source.size());

    std::vector<int>   idx(1);
    std::vector<float> sqd(1);

    for (const auto& p : source) {
        if (kdtree.nearestKSearch(p, 1, idx, sqd) > 0)
            dists.push_back(std::sqrt(static_cast<double>(sqd[0])));
        else
            dists.push_back(std::numeric_limits<double>::max());
    }
    return dists;
}

RegistrationResult evaluate_registration(
    const PointCloud& source,
    const PointCloud& target,
    double max_distance,
    const Eigen::Matrix4d& transformation) {

    PointCloud source_T = transform_point_cloud_copy(source, transformation);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target.makeShared());

    RegistrationResult res;
    res.transformation_ = transformation;

    double sum_sq = 0.0;
    std::vector<int>   idx(1);
    std::vector<float> sqd(1);

    for (int si = 0; si < static_cast<int>(source_T.size()); ++si) {
        if (kdtree.nearestKSearch(source_T[si], 1, idx, sqd) > 0) {
            double d = std::sqrt(static_cast<double>(sqd[0]));
            if (d <= max_distance) {
                res.correspondence_set_.emplace_back(si, idx[0]);
                sum_sq += sqd[0];
            }
        }
    }

    if (!source.empty())
        res.fitness_ = static_cast<double>(res.correspondence_set_.size()) /
                       static_cast<double>(source.size());

    if (!res.correspondence_set_.empty())
        res.inlier_rmse_ = std::sqrt(sum_sq / res.correspondence_set_.size());

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<double, std::vector<double>>
compute_rmse_between_point_clouds(
    const PointCloud& source,
    const PointCloud& target) {

    if (source.size() != target.size())
        throw std::runtime_error(
            "Point clouds must have the same number of points. Source: " +
            std::to_string(source.size()) + ", Target: " +
            std::to_string(target.size()));

    int n = static_cast<int>(source.size());
    std::vector<double> dists(n);
    for (int i = 0; i < n; ++i) {
        double dx = source[i].x - target[i].x;
        double dy = source[i].y - target[i].y;
        double dz = source[i].z - target[i].z;
        dists[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    double mse = 0.0;
    for (double d : dists) mse += d * d;
    mse /= n;
    double rmse = std::sqrt(mse);
    logger.debug("Computed RMSE = " + std::to_string(rmse));
    return {rmse, dists};
}

// ─────────────────────────────────────────────────────────────────────────────
double compute_rmse_transformations(
    const Eigen::Matrix4d& transf_est,
    const Eigen::Matrix4d& transf_gt,
    const PointCloud& pcd) {

    PointCloud pcd_est = transform_point_cloud_copy(pcd, transf_est);
    PointCloud pcd_gt  = transform_point_cloud_copy(pcd, transf_gt);
    auto [rmse, _] = compute_rmse_between_point_clouds(pcd_est, pcd_gt);
    return rmse;
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix4d load_gt_transform(const std::string& json_file) {
    if (!fs::exists(json_file)) {
        logger.error("Ground-truth file not found: " + json_file);
        return Eigen::Matrix4d::Identity();
    }
    try {
        std::ifstream ifs(json_file);
        json j;
        ifs >> j;

        auto H_json = j.at("H");
        Eigen::Matrix4d T;
        if (H_json.is_array() && H_json[0].is_array()) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    T(r, c) = H_json[r][c].get<double>();
        } else {
            for (int i = 0; i < 16; ++i)
                T(i / 4, i % 4) = H_json[i].get<double>();
        }

        logger.info("Ground-truth transform (m):\n" + [&]() {
            std::ostringstream oss; oss << T; return oss.str(); }());

        Eigen::Matrix4d T_mm = T;
        T_mm.block<3, 1>(0, 3) *= 1000.0;
        return T_mm;

    } catch (const std::exception& e) {
        logger.error("Failed to load GT file: " + std::string(e.what()));
        return Eigen::Matrix4d::Identity();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
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
    const std::string& output_dir) {

    logger.info(std::string(30, '-') + " Calculating Registration Error " +
                std::string(30, '-'));

    // ── Full-cloud distances ──────────────────────────────────────────────
    PointCloud source_T = transform_point_cloud_copy(source_raw, icp_sol.transformation_);
    auto distances = compute_point_cloud_distance(source_T, target_raw);

    double mean_dist = 0.0, std_dist = 0.0;
    for (double d : distances) mean_dist += d;
    mean_dist /= distances.size();
    for (double d : distances) std_dist += (d - mean_dist) * (d - mean_dist);
    std_dist = std::sqrt(std_dist / distances.size());
    double max_dist = *std::max_element(distances.begin(), distances.end());

    logger.debug("Mean distance (full cloud): " + std::to_string(mean_dist));
    logger.debug("Std dev distance           : " + std::to_string(std_dist));

    // ── TEASER++ inlier stats ─────────────────────────────────────────────
    auto translation_inliers = teaser_solver.getTranslationInliers();
    auto rotation_inliers    = teaser_solver.getRotationInliers();

    logger.info("TEASER++ Internal Metrics:");
    logger.info("  Translation inliers: " +
        std::to_string(translation_inliers.size()) + " / " +
        std::to_string(num_corrs) + " (" +
        std::to_string(100.0 * translation_inliers.size() / num_corrs) + "%)");
    logger.info("  Rotation inliers   : " +
        std::to_string(rotation_inliers.size()) + " / " +
        std::to_string(num_corrs));

    // ── PCL evaluation ────────────────────────────────────────────────────
    auto eval = evaluate_registration(source_raw, target_raw, noise_bound,
                                       icp_sol.transformation_);

    logger.info("Evaluation Metrics:");
    logger.info("  Fitness          : " + std::to_string(eval.fitness_));
    logger.info("  Inlier RMSE (mm) : " + std::to_string(eval.inlier_rmse_));
    logger.info("  Correspondence set: " +
        std::to_string(eval.correspondence_set_.size()) + " / " +
        std::to_string(source_raw.size()));
    logger.info("  FPFH correspondences: " + std::to_string(num_corrs));

    // ── Inlier point-to-point distance ────────────────────────────────────
    const auto& corr = icp_sol.correspondence_set_;
    PointCloud src_corr_pcd, tgt_corr_pcd;
    for (auto& [si, ti] : corr) {
        src_corr_pcd.push_back(source_raw[si]);
        tgt_corr_pcd.push_back(target_raw[ti]);
    }
    PointCloud src_corr_T = transform_point_cloud_copy(src_corr_pcd,
                                                        icp_sol.transformation_);
    auto inlier_dists = compute_point_cloud_distance(src_corr_T, tgt_corr_pcd);

    double mean_inlier_dist = 0.0;
    for (double d : inlier_dists) mean_inlier_dist += d;
    if (!inlier_dists.empty()) mean_inlier_dist /= inlier_dists.size();
    logger.debug("Mean inlier distance: " + std::to_string(mean_inlier_dist));

    // ── Correct ICP transform back to metres ─────────────────────────────
    Eigen::Matrix4d T_icp_m = icp_sol.transformation_;
    T_icp_m.block<3, 1>(0, 3) /= 1000.0;
    Eigen::Matrix4d T_icp_corrected_m = T_icp_m * trans_init;

    Eigen::Matrix4d T_icp_corrected = T_icp_corrected_m;
    T_icp_corrected.block<3, 1>(0, 3) *= 1000.0;

    logger.info("Estimated matrix (m):");
    { std::ostringstream oss; oss << T_icp_corrected_m; logger.info(oss.str()); }

    // ── Diagonal length for normalised RMSE ──────────────────────────────
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(target_raw, min_pt, max_pt);
    double diag = std::sqrt(
        std::pow(max_pt.x - min_pt.x, 2) +
        std::pow(max_pt.y - min_pt.y, 2) +
        std::pow(max_pt.z - min_pt.z, 2));
    double rmse_pct = icp_sol.inlier_rmse_ / diag * 100.0;
    logger.debug("Target diagonal length: " + std::to_string(diag) + " mm");
    logger.debug("RMSE % of diagonal    : " + std::to_string(rmse_pct));

    // ── Ground-truth comparison ───────────────────────────────────────────
    std::string source_json = args.source;
    for (auto& ext : {".ply", ".pcd"}) {
        size_t p = source_json.rfind(ext);
        if (p != std::string::npos) { source_json.replace(p, strlen(ext), ".json"); break; }
    }

    Eigen::Matrix4d gt = load_gt_transform(source_json);
    auto [rot_err, trans_err] = transformation_error(T_icp_corrected, gt);

    logger.info("Error vs Ground Truth:");
    logger.info("  Rotation  (rad/deg): " + std::to_string(rot_err) + " / " +
        std::to_string(rot_err * 180.0 / M_PI));
    logger.info("  Translation (mm)   : " + std::to_string(trans_err));

    double reg_rmse = compute_rmse_transformations(T_icp_corrected, gt, source_raw);
    logger.debug("Registration RMSE: " + std::to_string(reg_rmse) + " mm");

    // ── Percentages ───────────────────────────────────────────────────────
    double pct_to_tgt = static_cast<double>(corr.size()) / target_raw.size();
    double pct_to_src = static_cast<double>(corr.size()) / source_raw.size();

    logger.info("Timing:");
    logger.info("  Total registration time: " +
        std::to_string(registration_total_time) + " sec");

    // ── Save JSON ─────────────────────────────────────────────────────────
    json output;
    output["source"]            = args.source;
    output["target"]            = args.target;
    output["voxel_size"]        = args.voxel_size;
    output["refinement_voxel_size"] = args.refinement_voxel_size;
    output["rotation_error_rad"]  = rot_err;
    output["rotation_error_deg"]  = rot_err * 180.0 / M_PI;
    output["translation_error"]   = trans_err;
    output["fitness"]             = icp_sol.fitness_;
    output["inlier_rmse"]         = icp_sol.inlier_rmse_;
    output["rmse_percentage_to_target_diagonal"] = rmse_pct;
    output["mean_distance_points_full_cloud"]     = mean_dist;
    output["max_distance_points_full_cloud"]      = max_dist;
    output["standard_deviation_distance_full_cloud"] = std_dist;
    output["inlier_mean_distance"]       = mean_inlier_dist;
    output["registration_total_time_sec"] = registration_total_time;
    output["nb_of_points_target_down"]   = target_down_nb_points;
    output["nb_of_points_source_down"]   = source_down_nb_points;
    output["nb_of_fpfh_correspondences"] = num_corrs;
    output["percentage_inliers_to_target"] = pct_to_tgt;
    output["percentage_inliers_to_source"] = pct_to_src;

    std::vector<double> est_flat(16), prod_flat(16);
    Eigen::Map<Eigen::Matrix4d>(est_flat.data())  = T_icp_corrected;
    Eigen::Matrix4d prod = T_icp_corrected * gt;
    Eigen::Map<Eigen::Matrix4d>(prod_flat.data()) = prod;
    output["estimated_transformation"]       = est_flat;
    output["product_of_the_transformations"] = prod_flat;

    try {
        fs::create_directories(output_dir);
        fs::path src_path(args.source);
        std::string fname = src_path.stem().string() + "_metrics.json";
        fs::path out_file = fs::path(output_dir) / fname;
        std::ofstream ofs(out_file);
        ofs << output.dump(4);
        logger.info("Saved metrics to " + out_file.string());
    } catch (const std::exception& e) {
        logger.error("Failed to save metrics: " + std::string(e.what()));
    }
    logger.info(std::string(100, '-'));
}

// ─────────────────────────────────────────────────────────────────────────────
void calculate_errors(
    const RegistrationArgs& args,
    const RegistrationResult* icp_sol,
    const Eigen::Matrix4d& estimated_transform,
    double voxel_size,
    const std::string& scan_gt_json,
    double total_time,
    const PointCloud& source,
    const PointCloud& target,
    const std::string& output_dir) {

    logger.info(std::string(30, '-') + " Calculating Teaser++ Errors " +
                std::string(30, '-'));

    Eigen::Matrix4d gt = load_gt_transform(scan_gt_json);
    { std::ostringstream oss; oss << gt; logger.info("GT transform (mm):\n" + oss.str()); }

    auto [rot_err, trans_err] = transformation_error(estimated_transform, gt);
    { std::ostringstream oss; oss << "Product:\n" << (estimated_transform * gt);
      logger.info(oss.str()); }
    logger.info("Rotation (rad/deg): " + std::to_string(rot_err) + " / " +
        std::to_string(rot_err * 180.0 / M_PI) +
        "  Translation: " + std::to_string(trans_err) + " mm");

    double fitness     = std::numeric_limits<double>::quiet_NaN();
    double inlier_rmse = std::numeric_limits<double>::quiet_NaN();
    if (icp_sol) {
        fitness     = icp_sol->fitness_;
        inlier_rmse = icp_sol->inlier_rmse_;
    }

    // Full-cloud distances
    PointCloud src_T = transform_point_cloud_copy(source, estimated_transform);
    auto dists = compute_point_cloud_distance(src_T, target);

    double mean_d = 0.0, std_d = 0.0;
    for (double d : dists) mean_d += d;
    mean_d /= dists.size();
    for (double d : dists) std_d += (d - mean_d) * (d - mean_d);
    std_d = std::sqrt(std_d / dists.size());
    double max_d = *std::max_element(dists.begin(), dists.end());

    logger.info("Mean distance : " + std::to_string(mean_d));
    logger.info("Std  distance : " + std::to_string(std_d));

    json output;
    output["source"]            = args.source;
    output["target"]            = args.target;
    output["voxel_size"]        = voxel_size;
    output["rotation_error_rad"]  = rot_err;
    output["rotation_error_deg"]  = rot_err * 180.0 / M_PI;
    output["translation_error"]   = trans_err;
    output["fitness"]             = fitness;
    output["inlier_rmse"]         = inlier_rmse;
    output["mean_distance_points_full_cloud"] = mean_d;
    output["max_distance_points_full_cloud"]  = max_d;
    output["standard_deviation_distance_full_cloud"] = std_d;
    output["registration_total_time_sec"] = total_time;

    std::vector<double> est_flat(16);
    Eigen::Map<Eigen::Matrix4d>(est_flat.data()) = estimated_transform;
    output["estimated_transformation"] = est_flat;

    Eigen::Matrix4d prod = estimated_transform * gt;
    std::vector<double> prod_flat(16);
    Eigen::Map<Eigen::Matrix4d>(prod_flat.data()) = prod;
    output["product_of_the_transformations"] = prod_flat;

    try {
        fs::create_directories(output_dir);
        fs::path gt_path(scan_gt_json);
        std::string fname = gt_path.stem().string() + "_metrics.json";
        fs::path out_file = fs::path(output_dir) / fname;
        std::ofstream ofs(out_file);
        ofs << output.dump(4);
        logger.info("Saved metrics to " + out_file.string());
    } catch (const std::exception& e) {
        logger.error("Failed to save metrics: " + std::string(e.what()));
    }
    logger.info(std::string(100, '-'));
}

// ─────────────────────────────────────────────────────────────────────────────
void save_estimated_poses(
    const Eigen::Matrix4d& estimated_transform,
    const std::string& source_path,
    const std::string& output_dir) {

    std::vector<double> flat(16);
    Eigen::Map<Eigen::Matrix4d>(flat.data()) = estimated_transform;

    json output;
    output["H"] = flat;

    try {
        fs::create_directories(output_dir);
        fs::path src_path(source_path);
        std::string fname = src_path.stem().string() + ".json";
        fs::path out_file = fs::path(output_dir) / fname;
        std::ofstream ofs(out_file);
        ofs << output.dump(4);
        logger.info("Saved poses to " + out_file.string());
    } catch (const std::exception& e) {
        logger.error("Failed to save poses: " + std::string(e.what()));
    }
}

} // namespace registration::utils
