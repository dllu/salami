cmake_minimum_required(VERSION 3.10)
project(salami-slam)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)

set(SALAMI_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/src
	${PROJECT_SOURCE_DIR}/third-party
	${EIGEN3_INCLUDE_DIR}
    )

################################################################

# avx_sincos segfaults without -mavx2
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++14 -march=native -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

################################################################

include_directories(${SALAMI_INCLUDE_DIRS})

add_executable(
    salami
    src/main.cpp
)
set_property(TARGET salami PROPERTY CXX_STANDARD 14)

add_executable(
    salami_calibrator
    src/calibrator.cpp
)
set_property(TARGET salami_calibrator PROPERTY CXX_STANDARD 14)

add_executable(
    salami_loop
    src/loop.cpp
)
set_property(TARGET salami_loop PROPERTY CXX_STANDARD 14)

################################################################
# kitti evaluation tool

add_executable(
    kitti_eval
    ${PROJECT_SOURCE_DIR}/third-party/devkit/cpp/evaluate_odometry.cpp
    ${PROJECT_SOURCE_DIR}/third-party/devkit/cpp/matrix.cpp
    )
target_include_directories(
    kitti_eval PRIVATE
    ${PROJECT_SOURCE_DIR}/third-party/devkit/cpp/
    )
set_property(TARGET kitti_eval PROPERTY COMPILE_FLAGS "-O3 -DNDEBUG -Wno-error -Wno-all")
