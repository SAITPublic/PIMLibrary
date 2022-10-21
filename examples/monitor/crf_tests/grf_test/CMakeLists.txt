set(SOURCE grf_test.cpp)

cmake_minimum_required(VERSION 3.5.1)
add_executable(grf_test.out ${SOURCE})

target_link_libraries(grf_test.out)

