#include "cl_util.h"

#include <iostream>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

using namespace std::literals;

void print_devices() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (const auto& platform : platforms)	{
    std::cout << "Platform " << platform.getInfo<CL_PLATFORM_NAME>()
      << "; version: " << platform.getInfo<CL_PLATFORM_VERSION>()
      << "; vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << '\n';
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (const auto& device : devices) {
      auto get_work_item_sizes = [&]() -> std::string {
        std::string res = "[";
        for (const auto v : device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
          res += " " + std::to_string(v);
        }
        res += " ]";
        return res;
      };
      auto get_device_type = [&]() -> std::string {
        const auto device_type = device.getInfo<CL_DEVICE_TYPE>();
        constexpr int device_count = 3;
        constexpr std::array<std::pair<int, std::string_view>, device_count> types = {{
          { CL_DEVICE_TYPE_CPU, "CPU"sv },
          { CL_DEVICE_TYPE_GPU, "GPU"sv },
          { CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR"sv }
        }};
        std::string res;
        for (const auto [type, label] : types) {
          if (device_type & type) {
            res = label;
            break;
          }
        }
        if (res.empty()) {
          res = "UNKNOWN";
        }
        if (device_type & CL_DEVICE_TYPE_DEFAULT) {
          res = res + " [DEFAULT]";
        }
        return res;
      };
      auto get_cache_info = [&]() -> std::string {
        const auto cache_type = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();
        const auto cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
        const auto cache_line_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
        if (cache_type == CL_NONE) {
          return "none";
        }
        std::string res;
        res += std::to_string(cache_size) + " bytes";
        if (cache_type == CL_READ_ONLY_CACHE) {
          res += "; read-only";
        }
        if (cache_type == CL_READ_WRITE_CACHE) {
          res += "; read-write";
        }
        res += "; cache line size = " + std::to_string(cache_line_size) + " bytes";
        return res;
      };
      auto get_global_memory_info = [&]() -> std::string {
        const auto size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        const auto max_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        std::string res;
        res += std::to_string(size) + " bytes";
        res += "; max. allocation = " + std::to_string(max_alloc_size) + " bytes";
        return res;
      };
      auto get_local_memory_info = [&]() -> std::string {
        const auto type = device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
        const auto size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        std::string res;
        res += std::to_string(size) + " bytes";
        if (type == CL_LOCAL) {
          res += "; local (e.g. dedicated SRAM)";
        }
        if (type == CL_GLOBAL) {
          res += "; global";
        }
        return res;
      };
      std::cout << "\t" << get_device_type() << " " << device.getInfo<CL_DEVICE_NAME>() << '\n'
        << "\t\tVersion: " << device.getInfo<CL_DEVICE_VERSION>() << '\n'
        << "\t\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>() << '\n'
        << "\t\tCompute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << '\n'
        << "\t\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << '\n'
        << "\t\tMax work item sizes: " << get_work_item_sizes() << '\n'
        << "\t\tGlobal memory: " << get_global_memory_info() << '\n'
        << "\t\tGlobal cache: " << get_cache_info() << '\n'
        << "\t\tLocal memory: " << get_local_memory_info() << '\n';
    }
  }
}

std::string get_error_string(int code) {
  switch (code) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
    case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID: return "CL_INVALID_SPEC_ID";
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
  }
  return std::to_string(code);
}
