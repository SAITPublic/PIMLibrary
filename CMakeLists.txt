cmake_minimum_required(VERSION 2.6)
project(FIM_LIBRARY)

#Define minimum version for GCC
set(GCC_EXPECTED_VERSION 4.9.0)
if(GCC_EXPECTED_VERSION VERSION_GREATER CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "GCC: Fim-c++ requires atleast version ${GCC_EXPECTED_VERSION}")
endif()

set(FIM_LIBRARY_VERSION_MAJOR 1)
set(FIM_LIBRARY_VERSION_MINOR 0)

set(PREFIX ${CMAKE_INSTRALL_PREFIX})
set(EXEC_PREFIX "${PREFIX}/bin")
set(ROCM_PATH $ENV{ROCM_PATH})

# Set Default build type to "Debug"
if((NOT CMAKE_BUILD_TYPE) OR (CMAKE_BUILD_TYPE) STREQUAL None)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build Type is ${CMAKE_BUILD_TYPE}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BUILD_DIR})
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/hip/bin/hipcc")

# Set CMake Flags
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unused-command-line-argument")
set(CMAKE_C_FLAGS "-std=c11")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -DPROFILE -g -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG -DPROFILE -g -O0")
endif()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

set(HCC_CXX_FLAGS  "-hc --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include"
                    "${PROJECT_SOURCE_DIR}/tools/crfcodegen/include"
                    "${PROJECT_SOURCE_DIR}/tools/profiler/include"
                    "${ROCM_PATH}/include")
link_directories("${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${PROJECT_SOURCE_DIR}/external_libs")

add_subdirectory(runtime/source)
add_subdirectory(examples)
add_subdirectory(tools)
