cmake_minimum_required(VERSION 2.8)
project(DepthPipe)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -g -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g  -O3 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 

# Check C++17 or C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 or C++14 support. Please use a different C++ compiler.")
endif()

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA. Project will not be built.")
endif(NOT CUDA_FOUND)
message(STATUS "CUDA Version is ${CUDA_VERSION}")
message(STATUS "CUDA include dirs ${CUDA_INCLUDE_DIRS}")
# set(CUDA_ARCH "-arch=sm_72")  # Replace sm_75 with your architecture (e.g., sm_60, sm_80, etc.) sm_72 for jetson xavier nx
# set(CUDA_ARCH "-arch=sm_86")  # Replace sm_75 with your architecture (e.g., sm_60, sm_80, etc.) sm_72 for jetson xavier nx
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH})
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-lineinfo") # Enable line info for debugging

# Automatically detect and set CUDA architecture
include(FindCUDA)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)
message(STATUS "Setting CUDA arch to ${ARCH_FLAGS}.")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3")


find_package(OpenCV REQUIRED)
MESSAGE("OPENCV VERSION: ${OpenCV_VERSION}")
find_package(Eigen3 REQUIRED)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt-cpp-api/cmake" ${CMAKE_MODULE_PATH})
# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /usr/src/tensorrt)
endif()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(TensorRT REQUIRED)

find_package(fmt REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/Thirdparty/CudaSift
${PROJECT_SOURCE_DIR}/Thirdparty/DepthNet/include
${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt-cpp-api/src
${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt-cpp-api/include
${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt-cpp-api/include/interfaces
${OpenCV_INCLUDE_DIRS}
${CUDA_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
file(GLOB SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cc")
add_library(${PROJECT_NAME} SHARED
   ${SOURCE_FILES}
#    src/DepthPipe.cc
#    src/DepthAnything.cc
#    src/model.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${TensorRT_LIBRARIES}
${CUDA_cudadevrt_LIBRARY}
${CUDA_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/CudaSift/lib/libcudasift.so
${PROJECT_SOURCE_DIR}/Thirdparty/tensorrt-cpp-api/lib/libtensorrt_cpp_api.so
fmt::fmt
)

# Build tests

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples)

add_executable(lizard_test Examples/lizard_test.cc)
target_link_libraries(lizard_test ${PROJECT_NAME})