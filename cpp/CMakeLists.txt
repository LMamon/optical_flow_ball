cmake_minimum_required(VERSION 3.10)
project(optical_flow_ball_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRE ON)

find_package(openCV Required)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(optical_flow_ball main.cpp)
target_link_libraries(optical_flow_ball ${OpenCV_LIBS})