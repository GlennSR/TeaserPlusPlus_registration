/// point_cloud.cpp – mirrors registration/utils/point_cloud.py

#include "registration/utils/point_cloud.hpp"
#include "registration/utils/helpers.hpp"
#include "registration/utils/logging.hpp"

#include <teaser/fpfh.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

namespace registration::utils {

static auto logger = get_logger("point_cloud");

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Load a PLY file robustly.
/// Handles both float32 and float64 (Open3D-style) x/y/z fields by reading
/// the raw bytes from PCLPointCloud2 and casting manually.
static bool load_ply_robust(const std::string& path, PointCloud& cloud) {
    pcl::PCLPointCloud2 cloud2;
    if (pcl::io::loadPLYFile(path, cloud2) < 0) return false;
    if (cloud2.width * cloud2.height == 0) return false;

    // Find x, y, z field offsets and datatypes
    int x_off = -1, y_off = -1, z_off = -1;
    uint8_t x_type = 0, y_type = 0, z_type = 0;
    for (const auto& f : cloud2.fields) {
        if (f.name == "x") { x_off = static_cast<int>(f.offset); x_type = f.datatype; }
        if (f.name == "y") { y_off = static_cast<int>(f.offset); y_type = f.datatype; }
        if (f.name == "z") { z_off = static_cast<int>(f.offset); z_type = f.datatype; }
    }
    if (x_off < 0 || y_off < 0 || z_off < 0) return false;

    auto read_float = [](const uint8_t* ptr, int offset, uint8_t dtype) -> float {
        const uint8_t* p = ptr + offset;
        if (dtype == pcl::PCLPointField::FLOAT32) {
            float v; std::memcpy(&v, p, 4); return v;
        }
        if (dtype == pcl::PCLPointField::FLOAT64) {
            double v; std::memcpy(&v, p, 8); return static_cast<float>(v);
        }
        return 0.f;
    };

    uint32_t n = cloud2.width * cloud2.height;
    uint32_t step = cloud2.point_step;
    cloud.clear();
    cloud.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* row = cloud2.data.data() + i * step;
        pcl::PointXYZ p;
        p.x = read_float(row, x_off, x_type);
        p.y = read_float(row, y_off, y_type);
        p.z = read_float(row, z_off, z_type);
        cloud.push_back(p);
    }

    cloud.width  = n;
    cloud.height = 1;
    cloud.is_dense = cloud2.is_dense;
    return !cloud.empty();
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

static PointCloud voxel_downsample(const PointCloud& pcd, double voxel_size) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pcd.makeShared());
    vg.setLeafSize(static_cast<float>(voxel_size),
                   static_cast<float>(voxel_size),
                   static_cast<float>(voxel_size));
    PointCloud out;
    vg.filter(out);
    return out;
}

/// Scale all points by factor (simulate pcd.scale())
static void scale_cloud(PointCloud& pcd, double factor) {
    for (auto& p : pcd) {
        p.x = static_cast<float>(p.x * factor);
        p.y = static_cast<float>(p.y * factor);
        p.z = static_cast<float>(p.z * factor);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<PointCloud, teaser::FPFHCloudPtr>
preprocess_point_cloud(const PointCloud& pcd, double voxel_size) {
    logger.debug("Downsample with voxel size " + std::to_string(voxel_size));
    PointCloud pcd_down = voxel_downsample(pcd, voxel_size);

    logger.debug("Computing FPFH (normal_r=" + std::to_string(voxel_size * 2.0) +
                 ", feature_r=" + std::to_string(voxel_size * 5.0) + ")");
    auto fpfh = extract_fpfh(pcd_down, voxel_size);
    return {pcd_down, fpfh};
}

// ─────────────────────────────────────────────────────────────────────────────
PointCloud load_point_cloud(const std::string& ply_path,
                             double voxel_size,
                             bool /*estimate_normals*/) {
    PointCloud pcd;
    if (!load_ply_robust(ply_path, pcd))
        throw std::runtime_error("Cannot read (or empty) point cloud: " + ply_path);

    scale_cloud(pcd, 1000.0);  // metres → mm

    if (voxel_size > 0.0)
        pcd = voxel_downsample(pcd, voxel_size);

    return pcd;
}

PointCloud load_point_cloud(const PointCloud& src,
                             double voxel_size,
                             bool /*estimate_normals*/) {
    if (src.empty())
        throw std::runtime_error("Input point cloud is empty.");

    PointCloud pcd = src;
    scale_cloud(pcd, 1000.0);

    if (voxel_size > 0.0)
        pcd = voxel_downsample(pcd, voxel_size);

    return pcd;
}

// ─────────────────────────────────────────────────────────────────────────────
std::tuple<PointCloud, PointCloud>
load_point_clouds_files_for_refinement(
    const std::string& source_ply,
    const std::string& target_ply,
    double             voxel_size,
    const Eigen::Matrix4d& trans_init) {

    auto source_down = load_point_cloud(source_ply, voxel_size);
    transform_point_cloud(source_down, trans_init);
    auto target_down = load_point_cloud(target_ply, voxel_size);
    return {source_down, target_down};
}

std::tuple<PointCloud, PointCloud>
load_point_clouds_for_refinement(
    const PointCloud& source,
    const PointCloud& target,
    double            voxel_size,
    const Eigen::Matrix4d& trans_init) {

    auto source_down = load_point_cloud(source, voxel_size);
    transform_point_cloud(source_down, trans_init);
    auto target_down = load_point_cloud(target, voxel_size);
    return {source_down, target_down};
}

// ─────────────────────────────────────────────────────────────────────────────
void noise_gaussian(PointCloud& pcd, double std_dev) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, std_dev);
    for (auto& p : pcd) {
        p.x += static_cast<float>(dist(rng));
        p.y += static_cast<float>(dist(rng));
        p.z += static_cast<float>(dist(rng));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
double rough_scale_point_cloud(const PointCloud& pcd) {
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(pcd, min_pt, max_pt);
    double max_ext = std::max({static_cast<double>(max_pt.x - min_pt.x),
                               static_cast<double>(max_pt.y - min_pt.y),
                               static_cast<double>(max_pt.z - min_pt.z)});
    return std::pow(10.0, std::floor(std::log10(max_ext)));
}

double rough_scale_point_cloud_from_file(const std::string& ply_path) {
    PointCloud pcd;
    load_ply_robust(ply_path, pcd);
    return rough_scale_point_cloud(pcd);
}

// ─────────────────────────────────────────────────────────────────────────────
Eigen::Matrix4d align_centers(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& trans_init,
    const Eigen::Matrix4d& correction) {

    PointCloud src = transform_point_cloud_copy(source, correction);
    PointCloud tgt = transform_point_cloud_copy(target, correction);
    transform_point_cloud(src, trans_init);

    Eigen::Vector4f c_src_f, c_tgt_f;
    pcl::compute3DCentroid(src, c_src_f);
    pcl::compute3DCentroid(tgt, c_tgt_f);

    Eigen::Vector3d translation =
        c_tgt_f.head<3>().cast<double>() - c_src_f.head<3>().cast<double>();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = translation;
    return T;
}

Eigen::Matrix4d align_centers_from_files(
    const std::string& source_file,
    const std::string& target_file,
    const Eigen::Matrix4d& trans_init,
    const Eigen::Matrix4d& correction) {

    PointCloud src, tgt;
    load_ply_robust(source_file, src);
    load_ply_robust(target_file, tgt);
    return align_centers(src, tgt, trans_init, correction);
}

// ─────────────────────────────────────────────────────────────────────────────
PointCloud filter_points_far_from_center(const PointCloud& pcd,
                                          double max_distance) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(pcd, centroid);

    PointCloud out;
    out.reserve(pcd.size());
    for (const auto& p : pcd) {
        float dx = p.x - centroid[0];
        float dy = p.y - centroid[1];
        float dz = p.z - centroid[2];
        if (std::sqrt(dx*dx + dy*dy + dz*dz) <= max_distance)
            out.push_back(p);
    }
    out.width  = static_cast<uint32_t>(out.size());
    out.height = 1;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
void transform_point_cloud(PointCloud& pcd, const Eigen::Matrix4d& T) {
    Eigen::Matrix3f R = T.block<3, 3>(0, 0).cast<float>();
    Eigen::Vector3f t = T.block<3, 1>(0, 3).cast<float>();
    for (auto& p : pcd) {
        Eigen::Vector3f v(p.x, p.y, p.z);
        Eigen::Vector3f vt = R * v + t;
        p.x = vt[0]; p.y = vt[1]; p.z = vt[2];
    }
}

PointCloud transform_point_cloud_copy(const PointCloud& pcd,
                                       const Eigen::Matrix4d& T) {
    PointCloud out = pcd;
    transform_point_cloud(out, T);
    return out;
}

} // namespace registration::utils

