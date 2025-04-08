cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 23)

project(parallel-programming)

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
# Prohibit downloading of Google Test
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
FetchContent_MakeAvailable(benchmark)

add_subdirectory(dot_product)
add_subdirectory(matmul)
add_subdirectory(seam_carving)
