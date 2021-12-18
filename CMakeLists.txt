cmake_minimum_required(VERSION 3.20)
project(neural_nets_cpp)

set(CMAKE_CXX_STANDARD 14)

find_package(ROOT REQUIRED COMPONENTS Core Gui)

include_directories(matrix)
include_directories(neural_net)
include_directories(ROOT_INCLUDE_DIRS)

add_subdirectory(googletest)
add_subdirectory(google_test)
enable_testing()


set(matrix_dir
        matrix/matrix.h
        matrix/matrix.cpp)

set(neural_net_dir
        neural_net/layer.h
        neural_net/layer.cpp
        neural_net/neural_net.h
        neural_net/neural_net.cpp
        neural_net/linear_layer.h
        neural_net/linear_layer.cpp
        )

add_executable(neural_nets_cpp examples/simple_net.cpp
        ${matrix_dir}
        ${neural_net_dir})

add_executable(xor_example examples/xor.cpp
        ${matrix_dir}
        ${neural_net_dir})

add_executable(mnist examples/MNIST-try/mnist.cpp
        ${matrix_dir}
        ${neural_net_dir})

add_executable(mnist_test examples/MNIST-try/mnist_test.cpp
        ${matrix_dir}
        ${neural_net_dir})

set(executables neural_nets_cpp mnist)

target_link_libraries(mnist PUBLIC ROOT::Core ROOT::Gui)
target_link_libraries(neural_nets_cpp PUBLIC ROOT::Core ROOT::Gui)
target_link_libraries(
        xor_example
        PUBLIC ROOT::Core ROOT::Gui)
target_link_libraries(
        mnist_test
        PUBLIC ROOT::Core ROOT::Gui)
