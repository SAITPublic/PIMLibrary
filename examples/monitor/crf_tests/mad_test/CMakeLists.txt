set(SOURCE mad_test.cpp)

cmake_minimum_required(VERSION 3.5.1)

add_executable(mad_test.out ${SOURCE})

if(NVIDIA AND OPENCL)
    target_link_libraries(mad_test.out)
endif()
if(AMD)
    target_link_libraries(mad_test.out hsakmt)
endif()

