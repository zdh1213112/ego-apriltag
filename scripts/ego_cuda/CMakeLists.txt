cmake_minimum_required(VERSION 3.18)
project(ego_apriltag_cuda LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "" FORCE)
set(WITH_GFLAGS ON CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_UNWIND OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(ego_gpu
  GIT_REPOSITORY https://github.com/Team766/apriltags_cuda.git
  GIT_TAG 65366174d5e735bf6d71e34328b2a064e16bffb4)
FetchContent_Declare(ego_decode
  GIT_REPOSITORY https://github.com/cgpadwick/apriltag.git
  GIT_TAG 67c0f842a1c24fb6a7527eba77cd5c10135cf8e6)
FetchContent_Declare(glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG b33e3bad4c46c8a6345525fd822af355e5ef9446)
# Build only the detector, without ROS, OpenCV, robot networking or demos.
FetchContent_GetProperties(ego_gpu)
if(NOT ego_gpu_POPULATED)
  FetchContent_Populate(ego_gpu)
endif()
FetchContent_MakeAvailable(ego_decode glog)
find_package(CUDAToolkit REQUIRED)
add_library(ego_apriltag_cuda SHARED
  bridge.cpp
  ${ego_gpu_SOURCE_DIR}/src/apriltag_gpu.cu
  ${ego_gpu_SOURCE_DIR}/src/apriltag_detect.cu
  ${ego_gpu_SOURCE_DIR}/src/cuda_frc971.cu
  ${ego_gpu_SOURCE_DIR}/src/labeling_allegretti_2019_BKE.cu
  ${ego_gpu_SOURCE_DIR}/src/line_fit_filter.cu
  ${ego_gpu_SOURCE_DIR}/src/points.cu
  ${ego_gpu_SOURCE_DIR}/src/threshold.cu)
set_source_files_properties(bridge.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(ego_apriltag_cuda PRIVATE
  ${ego_gpu_SOURCE_DIR}/src ${ego_decode_SOURCE_DIR}/common)
target_link_libraries(ego_apriltag_cuda PRIVATE apriltag glog::glog CUDA::cudart)
target_compile_options(ego_apriltag_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--std=c++20>)
# The CPU extension and the GPU fork must keep their AprilTag symbols separate.
target_link_options(ego_apriltag_cuda PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL)
set_target_properties(ego_apriltag_cuda PROPERTIES
  CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
