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
set(BUILD_TOOLS_PATH $ENV{BUILD_TOOLS_PATH})

option(TARGET "Enable TARGET apps build" OFF)
option(PIM_COMPILER "Enable PIM Compiler" OFF)
option(ENABLE_COVERAGE "Enable Test Coverage (lcov)" OFF)
option(NVIDIA "Enable TARGET:NVIDIA apps build" OFF)
option(AMD "Enable TARGET:AMD apps build" OFF)
option(OPENCL "Enable Opencl ON/OFF" ON)
option(MOBILE "Enable Mobile ON/OFF" OFF)

if(MOBILE)
    set(CMAKE_SYSTEM_NAME  Android)
    set(OPENCL ON)
endif()

if(NVIDIA)
    set(OPENCL ON)
endif()

if(AMD)
    set(ROCM_PATH ${BUILD_TOOLS_PATH})
endif()

# Set Default build type to "Debug"
if((NOT CMAKE_BUILD_TYPE) OR (CMAKE_BUILD_TYPE) STREQUAL None)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build Type is ${CMAKE_BUILD_TYPE}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BUILD_DIR})
set(CMAKE_CXX_FLAGS "-std=c++14")

# Set CMake Flags
if(AMD)
    set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wno-unused-command-line-argument -Wno-invalid-constexpr")
endif()

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
        "${PROJECT_SOURCE_DIR}/external_libs/include/dramsim2"
        "${PROJECT_SOURCE_DIR}/external_libs/include/dramsim2/lib"
        "${PROJECT_SOURCE_DIR}/tools/crfcodegen/include"
        "${PROJECT_SOURCE_DIR}/tools/profiler/include"
        "${BUILD_TOOLS_PATH}/include")

link_directories("${PROJECT_SOURCE_DIR}/external_libs")

if(AMD)
    link_directories("${BUILD_TOOLS_PATH}/hip/lib" "${BUILD_TOOLS_PATH}/hsa/lib" "${BUILD_TOOLS_PATH}/libhsakmt/lib" "${BUILD_TOOLS_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")
    link_directories("${BUILD_TOOLS_PATH}/opencl/lib/x86_64/")
    include_directories("${BUILD_TOOLS_PATH}/opencl/include/")
    add_definitions(-DROCM_PATH="${BUILD_TOOLS_PATH}")
endif()


if(OPENCL)
    set(OPENCL_PATH $ENV{OPENCL_PATH})
    link_directories("${OPENCL_PATH}/lib/")
    include_directories("${OPENCL_PATH}/include/")

    # default OpenCL version 2.0
    add_definitions(-DCL_KERNEL_SOURCE_PATH="${PROJECT_SOURCE_DIR}/runtime/include/executor/ocl/")
    add_definitions(-DCL_KERNEL_INCLUDE_PATH="${PROJECT_SOURCE_DIR}/runtime/include/CL_KERNEL_INCLUDE_PATH")
    add_definitions(-DCL_KERNEL_BINARY_PATH="${OPENCL_PATH}/bin/")
    add_definitions(-DCL_KERNEL_BUILD_PATH="${PROJECT_BUILD_DIR}/runtime/source/")
    add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
    add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS)
    add_definitions(-DPIM_PATH="${BUILD_TOOLS_PATH}")
endif()

#for test vectors default data path
add_definitions(-DTEST_VECTORS_DATA="${PROJECT_SOURCE_DIR}/test_vectors/")

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

if(AMD)
	add_definitions(-DAMD=1)
else()
	add_definitions(-DAMD=0)
endif()

if(NVIDIA)
	add_definitions(-DNVIDIA=1)
else()
	add_definitions(-DNVIDIA=0)
endif()

if(OPENCL)
	add_definitions(-DOPENCL=1)
else()
	add_definitions(-DOPENCL=0)
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
