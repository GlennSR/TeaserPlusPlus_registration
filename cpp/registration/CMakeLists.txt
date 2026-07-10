cmake_minimum_required(VERSION 3.10)

# ── Registration sub-library ──────────────────────────────────────────────────
add_library(registration_lib
    utils/helpers.cpp
    utils/point_cloud.cpp
    utils/transforms.cpp
    utils/metrics.cpp
    visualization/viewer.cpp
)

target_include_directories(registration_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..   # exposes registration/ as include root
        ${PCL_INCLUDE_DIRS}
)

target_link_libraries(registration_lib
    PUBLIC
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        teaserpp::teaser_registration
        teaserpp::teaser_features
        teaserpp::teaser_io
        nlohmann_json::nlohmann_json
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

target_compile_features(registration_lib PUBLIC cxx_std_17)
