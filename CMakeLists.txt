cmake_minimum_required(VERSION 3.7.2)
project(lowrank)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

include_directories(/home/kailaix/.local/include/eigen3)

add_executable(main lowrank.cpp)
add_library(lowrank SHARED lowrank.cpp)
set_property(TARGET lowrank PROPERTY POSITION_INDEPENDENT_CODE ON)


