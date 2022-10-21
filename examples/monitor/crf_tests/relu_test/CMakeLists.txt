set(SOURCE relu_test.cpp)

cmake_minimum_required(VERSION 3.5.1)

add_executable(relu_test.out ${SOURCE})

if(NVIDIA AND OPENCL)
    target_link_libraries(relu_test.out)
endif()
if(AMD)
    target_link_libraries(relu_test.out hsakmt)
endif()

