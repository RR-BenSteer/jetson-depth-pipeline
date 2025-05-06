#!/bin/sh

INSTALL_PATH=/usr/local/lib/aarch64-linux-gnu/

CUDA_SIFT_LIB=libcudasift.so
cd Thirdparty/CudaSift
echo "Configuring and building Thirdparty/CudaSift ..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8
cd ..
cp lib/$CUDA_SIFT_LIB $INSTALL_PATH/$CUDA_SIFT_LIB

TENSORRT_LIB=libtensorrt_cpp_api.so
cd ../tensorrt-cpp-api
echo "Configuring and building Thirdparty/tensorrt-cpp-api ..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8
cd ..
cp lib/$TENSORRT_LIB $INSTALL_PATH/$TENSORRT_LIB