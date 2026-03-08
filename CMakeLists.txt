cmake_minimum_required(VERSION 3.25)
project(voronoi-stippling)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(raylib REQUIRED)
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_23)
target_include_directories(project_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(project_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -Wall -Wextra -Werror -Wpedantic -Wshadow -Wformat=2 -Wconversion -Wno-unused-parameter -march=native>
)

add_library(
    stippling_core
    src/level0_rejection.cpp
    src/level1_brute_force.cpp
    src/level2_spatial_grid.cpp
    src/level3_prefix_sum.cpp
    src/level4_quadtree.cpp
    src/level5_tiled.cpp
    src/level4_parallel.cpp
    src/level4_par_unseq.cpp
    src/level2_par_unseq.cpp
)
target_link_libraries(stippling_core PUBLIC project_options raylib Threads::Threads)
target_link_libraries(stippling_core PUBLIC tbb)

function(add_stippling_level target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE project_options raylib)
    string(TOUPPER ${target} target_upper)
    target_compile_definitions(${target} PRIVATE STIPPLING_BUILD_${target_upper}_MAIN=1)
endfunction()

add_stippling_level(level0 src/level0_rejection.cpp)
add_stippling_level(level1 src/level1_brute_force.cpp)
add_stippling_level(level2 src/level2_spatial_grid.cpp)
add_stippling_level(level3 src/level3_prefix_sum.cpp)
add_stippling_level(level4 src/level4_quadtree.cpp)
add_stippling_level(level5 src/level5_tiled.cpp)
add_stippling_level(level4_parallel src/level4_parallel.cpp)
target_link_libraries(level4_parallel PRIVATE Threads::Threads)
add_stippling_level(level4_par_unseq src/level4_par_unseq.cpp)
target_link_libraries(level4_par_unseq PRIVATE Threads::Threads tbb)
add_stippling_level(level2_par_unseq src/level2_par_unseq.cpp)
target_link_libraries(level2_par_unseq PRIVATE Threads::Threads tbb)

add_executable(voronoi_bench bench/benchmark_levels.cpp)
target_link_libraries(voronoi_bench PRIVATE stippling_core benchmark::benchmark benchmark::benchmark_main)

add_executable(weighted_cvd src/weighted_cvd.cpp)
target_link_libraries(weighted_cvd PRIVATE project_options raylib Threads::Threads tbb)
