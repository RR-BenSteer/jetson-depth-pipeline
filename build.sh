cd Thirdparty/CudaSift
echo "Configuring and building Thirdparty/CudaSift ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../tensorrt-cpp-api
echo "Configuring and building Thirdparty/tensorrt-cpp-api ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../../
echo "Configuring and building jetson-depth-pipeline ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8