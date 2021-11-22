cmake_minimum_required(VERSION 3.19)
project(tests)

set(CMAKE_CXX_STANDARD 14)


include_directories(${CMAKE_SOURCE_DIR}/matrix)
include_directories(${CMAKE_SOURCE_DIR}/neural_net)

set(matrix_dir
        ${CMAKE_SOURCE_DIR}/matrix/matrix_double.cpp
        ${CMAKE_SOURCE_DIR}/matrix/matrix_double.h
        ${CMAKE_SOURCE_DIR}/matrix/matrix.h
        ${CMAKE_SOURCE_DIR}/matrix/matrix.cpp
        )
set(neural_net_dir
        ${CMAKE_SOURCE_DIR}/neural_net/neural_net.h
        ${CMAKE_SOURCE_DIR}/neural_net/neural_net.cpp
        ${CMAKE_SOURCE_DIR}/neural_net/layer.h
        ${CMAKE_SOURCE_DIR}/neural_net/layer.cpp
        )
add_executable(GTests test2.cpp
        ${matrix_dir}
        ${neural_net_dir})


target_include_directories(GTests
        PUBLIC ${CMAKE_SOURCE_DIR}/googletest/googletest/include
        )

target_link_libraries(GTests
        gtest
        gtest_main
        )

add_test(NAME google_tests
        COMMAND google_tests)