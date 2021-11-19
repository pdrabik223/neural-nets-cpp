cmake_minimum_required(VERSION 3.20)
project(neural_nets_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(matrix)
set(matrix_dir
        matrix/matrix_double.cpp
        matrix/matrix_double.h)


add_executable(neural_nets_cpp  main.cpp
        ${matrix_dir})
