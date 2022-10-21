set(SOURCE gemv_pim.cpp)

cmake_minimum_required(VERSION 3.5.1)

add_executable(gemv_pim.out ${SOURCE})

if(NVIDIA AND OPENCL)
    target_link_libraries(gemv_pim.out)
endif()
if(AMD)
    target_link_libraries(gemv_pim.out hsakmt)
endif()
