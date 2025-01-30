cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0074 NEW)

project(lidar_centerpoint)

# Global compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
add_compile_options(-Wno-deprecated-declarations)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# CUDA setup
option(CUDA_VERBOSE "Verbose output of CUDA modules" OFF)
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if(CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  if(CUDA_VERBOSE)
    message("CUDA is available!")
    message("CUDA Libs: ${CUDA_LIBRARIES}")
    message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  endif()
  unset(CUDA_cublas_device_LIBRARY CACHE)
  set(CUDA_AVAIL ON)
else()
  message(FATAL_ERROR "CUDA NOT FOUND")
endif()

# TensorRT setup
option(TRT_AVAIL "TensorRT available" OFF)
find_library(NVINFER nvinfer)
find_library(NVONNXPARSER nvonnxparser)
if(NVINFER AND NVONNXPARSER)
  if(CUDA_VERBOSE)
    message("TensorRT is available!")
    message("NVINFER: ${NVINFER}")
    message("NVONNXPARSER: ${NVONNXPARSER}")
  endif()
  set(TRT_AVAIL ON)
else()
  message(FATAL_ERROR "TensorRT NOT FOUND")
endif()

# CUDNN setup
option(CUDNN_AVAIL "CUDNN available" OFF)
find_library(CUDNN_LIBRARY cudnn)
if(CUDNN_LIBRARY)
  if(CUDA_VERBOSE)
    message(STATUS "CUDNN is available!")
    message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  endif()
  set(CUDNN_AVAIL ON)
else()
  message(FATAL_ERROR "CUDNN NOT FOUND")
endif()

# Fetch tensorrt_common
include(FetchContent)
FetchContent_Declare(
  tensorrt_common
  GIT_REPOSITORY https://github.com/ga58lar/tensorrt_common.git
  GIT_TAG non-ros
)
FetchContent_MakeAvailable(tensorrt_common)

# Set include directories
include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${tensorrt_common_SOURCE_DIR}/include
)

# Build CUDA library
cuda_add_library(centerpoint_cuda_lib SHARED
  lib/postprocess/circle_nms_kernel.cu
  lib/postprocess/postprocess_kernel.cu
  lib/network/scatter_kernel.cu
  lib/preprocess/preprocess_kernel.cu
)

# Build main library
add_library(centerpoint_lib SHARED
  lib/centerpoint_trt.cpp
  lib/utils.cpp
  lib/network/network_trt.cpp
  lib/network/tensorrt_wrapper.cpp
  lib/preprocess/pointcloud_densification.cpp
  lib/preprocess/voxel_generator.cpp
)

target_include_directories(centerpoint_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${tensorrt_common_SOURCE_DIR}/include>
)

target_include_directories(centerpoint_lib SYSTEM PUBLIC
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(centerpoint_lib PUBLIC
  centerpoint_cuda_lib
  tensorrt_common
  ${NVINFER}
  ${NVONNXPARSER}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARY}
  yaml-cpp
  Eigen3::Eigen
  stdc++fs
)

# Build executable
add_executable(centerpoint_detector src/main.cpp)

target_include_directories(centerpoint_detector PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${tensorrt_common_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(centerpoint_detector PRIVATE
  centerpoint_lib
  centerpoint_cuda_lib
  tensorrt_common
  ${NVINFER}
  ${NVONNXPARSER}
  ${CUDA_LIBRARIES}
  ${CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARY}
  yaml-cpp
  Eigen3::Eigen
  stdc++fs
)

add_dependencies(centerpoint_detector
  centerpoint_lib
  centerpoint_cuda_lib
  tensorrt_common
)

# Installation
install(TARGETS centerpoint_detector centerpoint_lib centerpoint_cuda_lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)