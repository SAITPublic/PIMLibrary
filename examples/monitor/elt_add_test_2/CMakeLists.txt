set(SOURCE elt_add_pim.cpp)

cmake_minimum_required(VERSION 3.5.1)
add_executable(elt_add_pim.out ${SOURCE})

if(NVIDIA AND OPENCL)
    target_link_libraries(elt_add_pim.out)
endif()
if(AMD)
    target_link_libraries(elt_add_pim.out hsakmt)
endif()
