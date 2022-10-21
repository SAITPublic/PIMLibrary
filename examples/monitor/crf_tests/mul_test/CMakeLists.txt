set(SOURCE mul_test.cpp)

cmake_minimum_required(VERSION 3.5.1)

add_executable(mul_test.out ${SOURCE})
if(NVIDIA AND OPENCL)
    target_link_libraries(mul_test.out)
endif()
if(AMD)
    target_link_libraries(mul_test.out hsakmt)
endif()

