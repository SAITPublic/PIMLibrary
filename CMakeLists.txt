cmake_minimum_required(VERSION 3.5.1)

project(PIM_LIBRARY)
#Define minimum version for GCC
set(GCC_EXPECTED_VERSION 4.9.0)
if(GCC_EXPECTED_VERSION VERSION_GREATER CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "GCC: Pim-c++ requires atleast version ${GCC_EXPECTED_VERSION}")
endif()

set(PIM_LIBRARY_VERSION_MAJOR 1)
set(PIM_LIBRARY_VERSION_MINOR 0)

set(PREFIX ${CMAKE_INSTALL_PREFIX})
set(EXEC_PREFIX "${PREFIX}/bin")
set(ROCM_PATH $ENV{ROCM_PATH})

option(TARGET "Enable TARGET apps build" OFF)
option(RADEON7 "Enable TARGET:RADEON7 apps build" OFF)
option(MI50 "Enable TARGET:MI50 apps build" OFF)
option(PIM_COMPILER "Enable PIM Compiler" OFF)
option(ENABLE_COVERAGE "Enable Test Coverage (lcov)" OFF)

# Set Default build type to "Debug"
if((NOT CMAKE_BUILD_TYPE) OR (CMAKE_BUILD_TYPE) STREQUAL None)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build Type is ${CMAKE_BUILD_TYPE}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BUILD_DIR})

# Set CMake Flags
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wno-unused-command-line-argument -Wno-invalid-constexpr")
set(CMAKE_C_FLAGS "-std=c14")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRELEASE_PIM -DPROFILE -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DRELEASE_PIM -DPROFILE -O3")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_PIM -DPROFILE -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG_PIM -DPROFILE -g")
endif()

if(ENABLE_COVERAGE)
	message("Test coverage option has been enabled")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include"
                    "${PROJECT_SOURCE_DIR}/external_libs/include"
                    "${PROJECT_SOURCE_DIR}/tools/crfcodegen/include"
                    "${PROJECT_SOURCE_DIR}/tools/profiler/include"
                    "${ROCM_PATH}/include")
link_directories("${ROCM_PATH}/hip/lib" "${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${ROCM_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")
link_directories("${ROCM_PATH}/opencl/lib/x86_64/")
include_directories("${ROCM_PATH}/opencl/include/")

#for test vectors default data path
add_definitions(-DTEST_VECTORS_DATA="${PROJECT_SOURCE_DIR}/test_vectors/")
add_definitions(-DROCM_PATH="${ROCM_PATH}")

# default OpenCL version 2.0
add_definitions(-DCL_KERNEL_SOURCE_PATH="${PROJECT_SOURCE_DIR}/runtime/include/executor/ocl/")
add_definitions(-DCL_KERNEL_BINARY_PATH="${ROCM_PATH}/opencl/bin/")
add_definitions(-DCL_KERNEL_BUILD_PATH="${PROJECT_BUILD_DIR}/runtime/source/")
add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS)

if(TARGET)
    add_definitions(-DTARGET=1)
else()
    add_definitions(-DEMULATOR=1)
endif()
if(RADEON7)
    add_definitions(-DRADEON7=1)
endif()
if(MI50)
    add_definitions(-DMI50=1)
endif()
if(PIM_COMPILER)
	add_definitions(-DPIM_COMPILER_ENABLE=1)
else()
	add_definitions(-DPIM_COMPILER_ENABLE=0)
endif()

add_subdirectory(runtime)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(external_libs)

set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_VERSION_MAJOR "${PIM_LIBRARY_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PIM_LIBRARY_VERSION_MINOR}")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "Samsung Electronics")
set(CPACK_PACKAGE_CONTACT "hyeonsu.kim@samsung.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Hyeonsu Kim")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
include(CPack)
