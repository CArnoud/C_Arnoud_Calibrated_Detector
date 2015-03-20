cmake_minimum_required (VERSION 2.6)
SET(CMAKE_CXX_COMPILER g++)
project (C_Arnoud_Calibrated_Detector)
find_package( OpenCV REQUIRED )
add_executable(C_Arnoud_Calibrated_Detector 
./src/C_Arnoud_Calibrated_Detector.cpp 
./src/Detector.cpp
./src/ColorChannel.cpp
./src/GradientMagnitudeChannel.cpp
./src/Pyramid.cpp
./src/QuantizedGradientChannel.cpp
./src/ChannelFeatures.cpp
./src/Options.cpp
./src/BoundingBox.cpp
./src/utils.cpp)
include_directories("/usr/include/opencv2")
include_directories("/usr/include/opencv2/highgui")
target_link_libraries( C_Arnoud_Calibrated_Detector ${OpenCV_LIBS} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
set(CMAKE_BUILD_TYPE Release)