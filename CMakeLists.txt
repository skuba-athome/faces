cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
rosbuild_add_executable(control src/control.cpp)
rosbuild_add_executable(detect src/detect.cpp)
target_link_libraries(detect /opt/ros/fuerte/lib/libopencv_core.so)
target_link_libraries(detect /opt/ros/fuerte/lib/libopencv_imgproc.so)
target_link_libraries(detect /opt/ros/fuerte/lib/libopencv_highgui.so)
target_link_libraries(detect /opt/ros/fuerte/lib/libopencv_objdetect.so)
target_link_libraries(detect /opt/ros/fuerte/lib/libopencv_legacy.so)

rosbuild_add_library(surf_libs	src/integral.cpp
								src/utils.cpp
								src/fasthessian.cpp
								src/ipoint.cpp
								src/surf.cpp)
rosbuild_add_compile_flags(surf_libs "-fPIC")
target_link_libraries(surf_libs /opt/ros/fuerte/lib/libopencv_calib3d.so)
rosbuild_add_executable(people src/people.cpp)
target_link_libraries(people /opt/ros/fuerte/lib/libopencv_core.so)
target_link_libraries(people /opt/ros/fuerte/lib/libopencv_imgproc.so)
target_link_libraries(people /opt/ros/fuerte/lib/libopencv_highgui.so)
target_link_libraries(people /opt/ros/fuerte/lib/libopencv_objdetect.so)
target_link_libraries(people /opt/ros/fuerte/lib/libopencv_legacy.so)
target_link_libraries(people surf_libs)


