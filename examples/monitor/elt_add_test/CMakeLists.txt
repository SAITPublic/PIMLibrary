set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
set(HCC_CXX_FLAGS  "-hc --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

set(SOURCE elt_add_pim.cpp)

cmake_minimum_required(VERSION 3.5.1)

include_directories(/opt/rocm/include)
#link_directories(/usr/local/lib)

add_executable(elt_add_pim.out ${SOURCE})

target_link_libraries(elt_add_pim.out hsakmt)

