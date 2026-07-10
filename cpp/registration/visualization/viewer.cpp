/// viewer.cpp – mirrors registration/visualization/viewer.py

#include "registration/visualization/viewer.hpp"
#include "registration/utils/logging.hpp"
#include "registration/utils/point_cloud.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>

#include <sstream>

namespace registration::visualization {

static auto logger = get_logger("viewer");

// ─────────────────────────────────────────────────────────────────────────────
void draw_registration_result(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& transformation,
    const std::string& window_name,
    double size,
    const Eigen::Matrix4d& target_frame_trans,
    const Eigen::Matrix4d& source_frame_trans) {

    using namespace registration::utils;

    // Deep-copy and apply transformation to source
    PointCloud source_T = transform_point_cloud_copy(source, transformation);

    // Compute source centre for origin of source frame
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(source, min_pt, max_pt);
    float cx = (min_pt.x + max_pt.x) * 0.5f;
    float cy = (min_pt.y + max_pt.y) * 0.5f;
    float cz = (min_pt.z + max_pt.z) * 0.5f;

    // Viewer
    pcl::visualization::PCLVisualizer viewer(window_name);
    viewer.setBackgroundColor(0.1, 0.1, 0.1);

    // Source (yellow): RGB 255, 180, 0
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        src_color(source_T.makeShared(), 255, 180, 0);
    viewer.addPointCloud(source_T.makeShared(), src_color, "source");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

    // Target (cyan): RGB 0, 166, 237
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        tgt_color(target.makeShared(), 0, 166, 237);
    viewer.addPointCloud(target.makeShared(), tgt_color, "target");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    // World frame (largest)
    viewer.addCoordinateSystem(size * 1.5, "world_frame");

    // Target frame
    Eigen::Affine3f tgt_affine =
        Eigen::Affine3f(target_frame_trans.cast<float>());
    viewer.addCoordinateSystem(size, tgt_affine, "target_frame");

    // Source frame (slightly smaller) placed at source centre
    Eigen::Affine3f src_affine =
        Eigen::Affine3f(source_frame_trans.cast<float>());
    src_affine.pretranslate(Eigen::Vector3f(cx, cy, cz));
    viewer.addCoordinateSystem(size / 1.5f, src_affine, "source_frame");

    viewer.addText(window_name, 10, 10, 14, 1.0, 1.0, 1.0, "title");
    viewer.resetCamera();

    logger.info("Showing viewer: " + window_name + "  (press 'q' to close)");
    viewer.spin();
}

// ─────────────────────────────────────────────────────────────────────────────
void save_registration_result(
    const PointCloud& source,
    const PointCloud& target,
    const Eigen::Matrix4d& transformation,
    const std::string& window_name,
    double size) {

    logger.info("Showing registration result: " + window_name);
    draw_registration_result(source, target, transformation, window_name, size);
}

// ─────────────────────────────────────────────────────────────────────────────
void print_point_cloud_info(
    const PointCloud& pcd,
    const std::string& name) {

    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(pcd, min_pt, max_pt);

    float ex = max_pt.x - min_pt.x;
    float ey = max_pt.y - min_pt.y;
    float ez = max_pt.z - min_pt.z;

    std::ostringstream oss;
    oss << "Point Cloud '" << name << "':\n"
        << "\tNumber of points : " << pcd.size() << "\n"
        << "\tBB extent (xyz)  : [" << ex << ", " << ey << ", " << ez << "]\n"
        << "\tMin (xyz)        : [" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << "]\n"
        << "\tMax (xyz)        : [" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << "]";
    logger.debug(oss.str());
}

} // namespace registration::visualization
