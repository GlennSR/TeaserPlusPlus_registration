/// main.cpp – C++ equivalent of teaserpp_fpfh_full.py
///
/// Build:
///   mkdir -p teaser_cpp/build && cd teaser_cpp/build
///   cmake .. -DCMAKE_BUILD_TYPE=Release
///   make -j$(nproc)
///
/// Run:
///   ./teaser_registration --source scan.ply --target map.ply [options]

#include "registration/utils/helpers.hpp"
#include "registration/utils/logging.hpp"
#include "registration/utils/metrics.hpp"
#include "registration/utils/point_cloud.hpp"
#include "registration/utils/transforms.hpp"
#include "registration/visualization/viewer.hpp"

#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <teaser/fpfh.h>
#include <teaser/matcher.h>
#include <teaser/registration.h>
#include <nlohmann/json.hpp>

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs   = std::filesystem;
using json     = nlohmann::json;
using namespace registration::utils;
using namespace registration::visualization;

static auto logger = get_logger("main");

// ─── Argument struct ──────────────────────────────────────────────────────────
struct Args {
    std::string source;
    std::string target;
    int         start_index           = 0;
    int         end_index             = -1;   // -1 = process to end
    double      voxel_size            = 30.0;
    double      refinement_voxel_size = 0.0;  // 0 = use voxel_size
    int         max_iter_icp          = 30;
    bool        refine_registration   = false;
    bool        use_gicp              = false;
    std::string output;                        // empty = no output
    LogLevel    verbose               = LogLevel::INFO;
    bool        viz                   = false;
};

// ─── Simple CLI parser ────────────────────────────────────────────────────────
static Args parse_args(int argc, char* argv[]) {
    Args a;
    auto get_next = [&](int& i) -> std::string {
        if (i + 1 >= argc)
            throw std::runtime_error("Missing value for " + std::string(argv[i]));
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if      (flag == "--source")                 a.source = get_next(i);
        else if (flag == "--target")                 a.target = get_next(i);
        else if (flag == "--start-index")            a.start_index = std::stoi(get_next(i));
        else if (flag == "--end-index")              a.end_index   = std::stoi(get_next(i));
        else if (flag == "--voxel-size")             a.voxel_size  = std::stod(get_next(i));
        else if (flag == "--refinement-voxel-size")  a.refinement_voxel_size = std::stod(get_next(i));
        else if (flag == "--max_iter_icp")           a.max_iter_icp = std::stoi(get_next(i));
        else if (flag == "--refine-registration")    a.refine_registration = true;
        else if (flag == "--use-gicp")               a.use_gicp = true;
        else if (flag == "--output" || flag == "-o") a.output = get_next(i);
        else if (flag == "--viz")                    a.viz = (get_next(i) == "true");
        else if (flag == "--verbose" || flag == "-v")
            a.verbose = log_level_from_string(get_next(i));
        else {
            std::cerr << "[WARN] Unknown argument: " << flag << "\n";
        }
    }
    if (a.source.empty() || a.target.empty())
        throw std::runtime_error("--source and --target are required.");
    return a;
}

// ─── prepare_dataset ─────────────────────────────────────────────────────────
struct Dataset {
    PointCloud           source_raw;
    PointCloud           target_raw;
    PointCloud           source_down;
    PointCloud           target_down;
    teaser::FPFHCloudPtr source_fpfh;
    teaser::FPFHCloudPtr target_fpfh;
};

static Dataset prepare_dataset(
    PointCloud source,
    PointCloud target,
    double voxel_size,
    const Eigen::Matrix4d& trans_init = Eigen::Matrix4d::Identity()) {

    transform_point_cloud(source, trans_init);

    logger.debug("Preprocessing source point cloud");
    auto [src_down, src_fpfh] = preprocess_point_cloud(source, voxel_size);
    registration::utils::print_point_cloud_info(src_down, "Downsampled source");

    logger.debug("Preprocessing target point cloud");
    auto [tgt_down, tgt_fpfh] = preprocess_point_cloud(target, voxel_size);
    registration::utils::print_point_cloud_info(tgt_down, "Downsampled target");

    return {std::move(source), std::move(target),
            std::move(src_down), std::move(tgt_down),
            std::move(src_fpfh), std::move(tgt_fpfh)};
}

// ─── refine_registration ─────────────────────────────────────────────────────
static RegistrationResult
refine_registration(
    const PointCloud& source,
    const PointCloud& target,
    double distance_threshold,
    const Eigen::Matrix4d& initial_transformation,
    int max_iteration,
    bool use_gicp = false) {

    std::string algo = use_gicp ? "Generalized ICP (GICP)" : "ICP";
    logger.debug("Pairwise " + algo + " registration, threshold=" +
                 std::to_string(distance_threshold));

    // Guard: both clouds need enough points for ICP/GICP to run
    const int MIN_PTS = 5;
    if (static_cast<int>(source.size()) < MIN_PTS ||
        static_cast<int>(target.size()) < MIN_PTS) {
        logger.warning(algo + " skipped: too few points (src=" +
            std::to_string(source.size()) + " tgt=" +
            std::to_string(target.size()) + ")");
        RegistrationResult empty;
        empty.transformation_ = initial_transformation;
        return empty;
    }

    Eigen::Matrix4f T_init = initial_transformation.cast<float>();
    Eigen::Matrix4f T_out  = Eigen::Matrix4f::Identity();

    if (use_gicp) {
        // k_correspondences must be < min(source, target) size
        int k_corr = std::min({20,
                               static_cast<int>(source.size()) - 1,
                               static_cast<int>(target.size()) - 1});
        k_corr = std::max(k_corr, 1);

        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(source.makeShared());
        gicp.setInputTarget(target.makeShared());
        gicp.setMaxCorrespondenceDistance(distance_threshold);
        gicp.setMaximumIterations(max_iteration);
        gicp.setTransformationEpsilon(1e-6);
        gicp.setCorrespondenceRandomness(k_corr);
        PointCloud aligned;
        gicp.align(aligned, T_init);
        T_out = gicp.getFinalTransformation();
    } else {
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(source.makeShared());
        icp.setInputTarget(target.makeShared());
        icp.setMaxCorrespondenceDistance(distance_threshold);
        icp.setMaximumIterations(max_iteration);
        icp.setTransformationEpsilon(1e-6);
        PointCloud aligned;
        icp.align(aligned, T_init);
        T_out = icp.getFinalTransformation();
    }

    Eigen::Matrix4d T_out_d = T_out.cast<double>();
    RegistrationResult res = evaluate_registration(
        source, target, distance_threshold, T_out_d);
    res.transformation_ = T_out_d;
    return res;
}

// ─── teaserpp_registration ────────────────────────────────────────────────────
struct TeaserResult {
    Eigen::Matrix4d T_teaser;
    std::unique_ptr<teaser::RobustRegistrationSolver> solver;
    int             target_down_nb_points;
    int             source_down_nb_points;
    int             num_corrs;
};

static TeaserResult teaserpp_registration(
    PointCloud source_raw,
    const Eigen::Matrix4d& trans_init,
    PointCloud target_raw,
    const Eigen::Matrix4d& target_toworld_transform,
    double voxel_size,
    bool visualize) {

    double frame_size = rough_scale_point_cloud(target_raw) / 7.5;

    // ── Initial visualisation ──────────────────────────────────────────────
    if (visualize) {
        draw_registration_result(source_raw, target_raw,
            Eigen::Matrix4d::Identity(),
            "Initial State (Source: Yellow, Target: Cyan)",
            frame_size, target_toworld_transform);
    }

    logger.debug([&] {
        std::ostringstream oss; oss << "Initial transformation:\n" << trans_init;
        return oss.str();
    }());

    if (visualize) {
        draw_registration_result(source_raw, target_raw, trans_init,
            "Corrected settings", frame_size,
            target_toworld_transform, trans_init);
    }

    // ── Preprocessing ──────────────────────────────────────────────────────
    auto ds = prepare_dataset(source_raw, target_raw, voxel_size, trans_init);
    int src_nb = static_cast<int>(ds.source_down.size());
    int tgt_nb = static_cast<int>(ds.target_down.size());

    if (visualize) {
        draw_registration_result(ds.source_down, ds.target_down,
            Eigen::Matrix4d::Identity(),
            "Downsampled Point Clouds", frame_size);
    }

    // ── Feature correspondences ────────────────────────────────────────────
    Eigen::MatrixXd source_xyz = pcd2xyz(ds.source_down);  // 3×N
    Eigen::MatrixXd target_xyz = pcd2xyz(ds.target_down);  // 3×M

    teaser::PointCloud src_teaser = pcl_to_teaser(ds.source_down);
    teaser::PointCloud tgt_teaser = pcl_to_teaser(ds.target_down);

    auto [corrs_A, corrs_B] = find_correspondences(
        src_teaser, tgt_teaser,
        *ds.source_fpfh, *ds.target_fpfh);

    int num_corrs = static_cast<int>(corrs_A.size());
    logger.debug("FPFH generates " + std::to_string(num_corrs) +
                 " putative correspondences.");

    Eigen::MatrixXd source_corr(3, num_corrs);
    Eigen::MatrixXd target_corr(3, num_corrs);
    for (int i = 0; i < num_corrs; ++i) {
        source_corr.col(i) = source_xyz.col(corrs_A[i]);
        target_corr.col(i) = target_xyz.col(corrs_B[i]);
    }

    // ── TEASER++ registration ──────────────────────────────────────────────
    double noise_bound = voxel_size * 2.0;
    auto solver = get_teaser_solver(noise_bound);
    solver->solve(source_corr, target_corr);

    auto sol = solver->getSolution();
    Eigen::Matrix4d T_teaser = Rt2T(sol.rotation, sol.translation);

    if (visualize) {
        draw_registration_result(ds.source_raw, ds.target_raw, T_teaser,
            "TEASER++ Registration Results", frame_size,
            target_toworld_transform, T_teaser * trans_init);
    }

    return {T_teaser, std::move(solver), tgt_nb, src_nb, num_corrs};
}

// ─── main registration pipeline ───────────────────────────────────────────────
static void run(Args& args) {
    // ── Load point clouds ──────────────────────────────────────────────────
    // Use PCLPointCloud2 + manual field extraction so double-precision PLY
    // files (Open3D default: "property double x/y/z") load correctly.
    auto load_ply = [](const std::string& path, PointCloud& cloud) -> bool {
        pcl::PCLPointCloud2 c2;
        if (pcl::io::loadPLYFile(path, c2) < 0) return false;
        if (c2.width * c2.height == 0) return false;

        int xo = -1, yo = -1, zo = -1;
        uint8_t xt = 0, yt = 0, zt = 0;
        for (const auto& f : c2.fields) {
            if (f.name == "x") { xo = (int)f.offset; xt = f.datatype; }
            if (f.name == "y") { yo = (int)f.offset; yt = f.datatype; }
            if (f.name == "z") { zo = (int)f.offset; zt = f.datatype; }
        }
        if (xo < 0 || yo < 0 || zo < 0) return false;

        auto rf = [](const uint8_t* p, int off, uint8_t dt) -> float {
            const uint8_t* q = p + off;
            if (dt == pcl::PCLPointField::FLOAT32) { float v; std::memcpy(&v,q,4); return v; }
            if (dt == pcl::PCLPointField::FLOAT64) { double v; std::memcpy(&v,q,8); return (float)v; }
            return 0.f;
        };

        uint32_t n = c2.width * c2.height, step = c2.point_step;
        cloud.clear(); cloud.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t* row = c2.data.data() + i * step;
            cloud.push_back({rf(row,xo,xt), rf(row,yo,yt), rf(row,zo,zt)});
        }
        cloud.width = n; cloud.height = 1; cloud.is_dense = c2.is_dense;
        return true;
    };

    PointCloud source_raw, target_raw;
    if (!load_ply(args.source, source_raw))
        throw std::runtime_error("Cannot read source (or empty): " + args.source);
    if (!load_ply(args.target, target_raw))
        throw std::runtime_error("Cannot read target (or empty): " + args.target);

    // Scale to mm
    auto scale_mm = [](PointCloud& pcd) {
        for (auto& p : pcd) { p.x *= 1000.f; p.y *= 1000.f; p.z *= 1000.f; }
    };
    scale_mm(source_raw);
    scale_mm(target_raw);

    // Diagonal lengths
    auto diag_len = [](const PointCloud& p) {
        pcl::PointXYZ mn, mx;
        pcl::getMinMax3D(p, mn, mx);
        return std::sqrt(std::pow(mx.x-mn.x,2) +
                         std::pow(mx.y-mn.y,2) +
                         std::pow(mx.z-mn.z,2));
    };
    logger.info("Target diagonal: " + std::to_string(diag_len(target_raw)) + " mm");
    logger.info("Source diagonal: " + std::to_string(diag_len(source_raw)) + " mm");

    double voxel_size = args.voxel_size;

    auto start_time = std::chrono::steady_clock::now();

    // ── Load initial guess from JSON ───────────────────────────────────────
    std::string pose_suffix = ".json";
    std::string init_guess_file = args.source;
    for (auto& ext : {".ply", ".pcd"}) {
        size_t p = init_guess_file.rfind(ext);
        if (p != std::string::npos) {
            init_guess_file.replace(p, strlen(ext), pose_suffix);
            break;
        }
    }
    logger.info("Initial guess file: " + init_guess_file);

    Eigen::Matrix4d trans_init = Eigen::Matrix4d::Identity();
    try {
        std::ifstream ifs(init_guess_file);
        if (!ifs.is_open()) throw std::runtime_error("not found");
        json j; ifs >> j;
        auto H = j.at("H");
        if (H[0].is_array()) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    trans_init(r, c) = H[r][c].get<double>();
        } else {
            for (int i = 0; i < 16; ++i)
                trans_init(i / 4, i % 4) = H[i].get<double>();
        }
    } catch (...) {
        logger.warning("Initial guess file not found: " + init_guess_file +
            " — using identity. Source scan roll/pitch will NOT be corrected;"
            " the clouds may appear tilted relative to each other.");
    }

    // ── Keep only roll+pitch from trans_init (zero out yaw) ───────────────
    // Mirrors Python: r = Rotation.from_matrix(R).as_euler('xyz'); r[2]=0
    // scipy 'xyz' (lowercase=extrinsic) = intrinsic ZYX: R = Rz(yaw)*Ry(pitch)*Rx(roll)
    // After zeroing yaw: R_new = Ry(pitch) * Rx(roll)
    {
        Eigen::Matrix3d R = trans_init.block<3, 3>(0, 0);

        // Standard ZYX decomposition (same range as scipy: pitch in [-π/2, π/2])
        double roll  = std::atan2( R(2,1), R(2,2));
        double pitch = std::atan2(-R(2,0), std::sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));
        // yaw = atan2(R(1,0), R(0,0)) — zeroed out

        Eigen::Matrix3d R_new =
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix()
          * Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX()).toRotationMatrix();

        trans_init.block<3, 3>(0, 0) = R_new;
        trans_init.block<3, 1>(0, 3).setZero();
    }
    logger.info([&] {
        std::ostringstream oss; oss << "Source initial guess:\n" << trans_init;
        return oss.str();
    }());

    // ── Load target ground-truth transform ────────────────────────────────
    Eigen::Matrix4d target_toworld = Eigen::Matrix4d::Identity();
    {
        std::string gt_file = args.target;
        for (auto& ext : {".ply", ".pcd"}) {
            size_t p = gt_file.rfind(ext);
            if (p != std::string::npos) {
                gt_file.replace(p, strlen(ext), "_gt_transform.json");
                break;
            }
        }
        if (!fs::exists(gt_file)) {
            gt_file = args.target;
            for (auto& ext : {".ply", ".pcd"}) {
                size_t p = gt_file.rfind(ext);
                if (p != std::string::npos) {
                    gt_file.replace(p, strlen(ext), ".json");
                    break;
                }
            }
        }
        try {
            std::ifstream ifs(gt_file);
            if (!ifs.is_open()) throw std::runtime_error("not found");
            json j; ifs >> j;
            auto H = j.at("H");
            if (H[0].is_array()) {
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        target_toworld(r, c) = H[r][c].get<double>();
            } else {
                for (int i = 0; i < 16; ++i)
                    target_toworld(i / 4, i % 4) = H[i].get<double>();
            }
            logger.debug([&] {
                std::ostringstream oss;
                oss << "Target GT transform:\n" << target_toworld;
                return oss.str();
            }());
        } catch (...) {
            logger.warning("Target GT transform not found, using identity.");
        }
    }
    transform_point_cloud(target_raw, target_toworld);

    // ── Remove outlier source points far from centre ───────────────────────
    source_raw = filter_points_far_from_center(source_raw, 20000.0);

    // ── TEASER++ global registration ──────────────────────────────────────
    auto [T_teaser, teaser_solver, tgt_nb, src_nb, num_corrs] =
        teaserpp_registration(source_raw, trans_init, target_raw,
                              target_toworld, voxel_size, args.viz);

    {
        std::ostringstream oss;
        oss << "\n\nEstimated TEASER++ transformation (mm):\n"
            << (T_teaser * trans_init) << "\n";
        logger.info(oss.str());
    }

    // Convert translation to metres for saving
    Eigen::Matrix4d T_teaser_m = T_teaser;
    T_teaser_m.block<3, 1>(0, 3) /= 1000.0;

    double teaser_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();

    if (!args.output.empty()) {
        std::string est_dir = args.output +
            "/Voxel" + std::to_string(static_cast<int>(voxel_size)) +
            "/Teaser_estimated_poses_" +
            std::to_string(static_cast<int>(voxel_size)) + "/";
        save_estimated_poses(T_teaser_m * trans_init, args.source, est_dir);

        std::string metrics_dir = args.output +
            "/Voxel" + std::to_string(static_cast<int>(voxel_size)) +
            "/Teaser_metrics_" +
            std::to_string(static_cast<int>(voxel_size)) + "/";

        RegistrationArgs ra{args.source, args.target, voxel_size};
        calculate_errors(ra, nullptr, T_teaser * trans_init, voxel_size,
                         init_guess_file, teaser_time,
                         source_raw, target_raw, metrics_dir);
    }

    // ── Optional ICP refinement ────────────────────────────────────────────
    if (args.refine_registration) {
        double ref_vox = (args.refinement_voxel_size > 0.0)
                          ? args.refinement_voxel_size
                          : voxel_size;
        logger.debug("Loading scan at refinement resolution (" +
                     std::to_string(ref_vox) + " mm)...");

        auto [source_ref, target_ref] = load_point_clouds_files_for_refinement(
            args.source, args.target, ref_vox, trans_init);

        auto icp_sol = refine_registration(
            source_ref, target_ref, ref_vox * 2.0,
            T_teaser, args.max_iter_icp, args.use_gicp);

        Eigen::Matrix4d T_icp = icp_sol.transformation_;
        {
            std::ostringstream oss;
            oss << "Refined transformation (ICP/GICP):\n" << T_icp;
            logger.debug(oss.str());
        }

        if (args.viz) {
            draw_registration_result(source_raw, target_raw, T_icp * trans_init,
                "ICP Refinement", 1.0,
                target_toworld, T_icp * trans_init);
        }

        double total_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();

        if (!args.output.empty()) {
            std::string reg_dir = args.output +
                "/Voxel" + std::to_string(static_cast<int>(voxel_size)) +
                "/Ref_Voxel" +
                std::to_string(static_cast<int>(args.refinement_voxel_size));

            RegistrationArgs ra{args.source, args.target, voxel_size,
                                args.refinement_voxel_size};
            registration_metrics(target_raw, source_raw, tgt_nb, src_nb,
                                  *teaser_solver, icp_sol, trans_init,
                                  num_corrs, voxel_size * 2.0, total_time,
                                  ra, reg_dir);

            Eigen::Matrix4d T_icp_m = T_icp;
            T_icp_m.block<3, 1>(0, 3) /= 1000.0;
            std::string poses_dir = reg_dir + "/Estimated_Poses/";
            save_estimated_poses(T_icp_m * trans_init, args.source, poses_dir);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        std::cerr << "Usage: teaser_registration --source FILE --target FILE [OPTIONS]\n";
        std::cerr << "  --voxel-size N            (default 30)\n";
        std::cerr << "  --refine-registration     (enable ICP refinement)\n";
        std::cerr << "  --use-gicp                (use GICP instead of ICP)\n";
        std::cerr << "  --refinement-voxel-size N (default = voxel-size)\n";
        std::cerr << "  --max_iter_icp N          (default 30)\n";
        std::cerr << "  --output/-o DIR           (output directory)\n";
        std::cerr << "  --verbose DEBUG|INFO|WARNING|ERROR|CRITICAL\n";
        std::cerr << "  --viz true                (enable PCL visualisation)\n";
        return 1;
    }

    // Initialise logging
    std::string log_file;
    if (!args.output.empty()) {
        fs::create_directories(args.output);
        log_file = args.output + "/Voxel" +
                   std::to_string(static_cast<int>(args.voxel_size)) + ".log";
    }
    setup_logging(args.verbose, log_file, "w");

    logger.info("Input arguments:");
    logger.info("  source            : " + args.source);
    logger.info("  target            : " + args.target);
    logger.info("  voxel_size        : " + std::to_string(args.voxel_size));
    logger.info("  refine_registration: " +
        std::string(args.refine_registration ? "true" : "false"));
    logger.info("  use_gicp          : " +
        std::string(args.use_gicp ? "true" : "false"));

    if (fs::is_directory(args.source)) {
        // Batch mode: run over all .ply / .pcd files in the directory
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(args.source)) {
            auto ext = entry.path().extension().string();
            if (ext == ".ply" || ext == ".pcd")
                files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
            return fs::path(a).stem().string() < fs::path(b).stem().string();
        });

        int end = (args.end_index < 0)
                  ? static_cast<int>(files.size())
                  : std::min(args.end_index, static_cast<int>(files.size()));

        int count = 1;
        for (int idx = args.start_index; idx < end; ++idx) {
            args.source = files[idx];
            logger.info("TEASER++ registration: " + args.source +
                        "  (" + std::to_string(count++) + "/" +
                        std::to_string(files.size()) + ")");
            try {
                run(args);
            } catch (const std::exception& e) {
                logger.error("Error processing " + args.source + ": " + e.what());
            }
        }
    } else {
        try {
            run(args);
        } catch (const std::exception& e) {
            logger.error("Registration failed: " + std::string(e.what()));
            return 1;
        }
    }

    return 0;
}
