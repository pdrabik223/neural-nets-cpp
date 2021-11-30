cmake_minimum_required(VERSION 3.20)
project(neural_nets_cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(matrix)
include_directories(neural_net)
add_subdirectory(googletest)
add_subdirectory(google_test)
enable_testing()

set(matrix_dir
        matrix/matrix_double.cpp
        matrix/matrix_double.h
        matrix/matrix.h
        matrix/matrix.cpp
        matrix/matrix_vector_operations.h
        )
set(neural_net_dir
        neural_net/neural_net.h
        neural_net/neural_net.cpp
        neural_net/layer.h
        neural_net/layer.cpp
        )
add_executable(neural_nets_cpp  main.cpp
        ${matrix_dir}
        ${neural_net_dir})
