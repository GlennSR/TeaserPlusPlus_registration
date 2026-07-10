cmake_minimum_required(VERSION 3.10)
project(teaserpp_registration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# ── Eigen3 ──────────────────────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ── PCL ──────────────────────────────────────────────────────────────────────
find_package(PCL 1.9 REQUIRED COMPONENTS
    common io filters features kdtree registration visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ── OpenMP (optional, used by PCL features) ───────────────────────────────────
find_package(OpenMP QUIET)

# ── TEASER++ ─────────────────────────────────────────────────────────────────
# Look for TEASER++ cmake exports produced by the sibling build directory.
set(TEASERPP_BUILD_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../TEASER-plusplus/build"
    CACHE PATH "Path to TEASER++ build directory")

find_package(teaserpp QUIET
    PATHS "${TEASERPP_BUILD_DIR}" "$ENV{HOME}/.local/lib/cmake/teaserpp"
    NO_DEFAULT_PATH)
if(NOT teaserpp_FOUND)
    find_package(teaserpp REQUIRED)
endif()

# ── nlohmann/json ─────────────────────────────────────────────────────────────
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via find_package – using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── Registration sub-library ──────────────────────────────────────────────────
add_subdirectory(registration)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        registration_lib
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        teaserpp::teaser_registration
        teaserpp::teaser_features
        teaserpp::teaser_io
        nlohmann_json::nlohmann_json
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
