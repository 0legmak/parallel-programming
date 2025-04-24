cmake_minimum_required(VERSION 3.10)

project(lock_free_ring_buffer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME}_test validate.cpp)
add_executable(${PROJECT_NAME}_bench bench.cpp)

target_link_libraries(${PROJECT_NAME}_bench benchmark::benchmark)
