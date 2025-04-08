
cmake_minimum_required(VERSION 3.11)

project(matmul LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} cl_util.cpp init.cpp solution.cpp validate.cpp)

add_executable(${PROJECT_NAME}_bench cl_util.cpp init.cpp solution.cpp bench.cpp)

target_link_libraries(${PROJECT_NAME}_bench benchmark::benchmark)

foreach(prog ${PROJECT_NAME} ${PROJECT_NAME}_bench)
  target_link_libraries(${prog} OpenCL::OpenCL)
  target_compile_definitions(${prog} PRIVATE
    CL_HPP_MINIMUM_OPENCL_VERSION=110
    CL_HPP_TARGET_OPENCL_VERSION=110
    CL_HPP_ENABLE_EXCEPTIONS=1
  )
endforeach(prog)
