set(SOURCE add_test.cpp)

cmake_minimum_required(VERSION 3.5.1)

add_executable(add_test.out ${SOURCE})

target_link_libraries(add_test.out)

