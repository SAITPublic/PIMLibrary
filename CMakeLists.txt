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

option(MIOPEN_APPS "Enable MIOpens apps build" OFF)
option(PYTORCH_BUILD "Enable pytorch custom operations build" OFF)
option(TF_BUILD "Enable tf custom operations build" OFF)
option(TARGET "Enable TARGET apps build" OFF)
option(RADEON7 "Enable TARGET:RADEON7 apps build" OFF)
option(MI50 "Enable TARGET:MI50 apps build" OFF)
option(ROCM3 "Enable Rocm3 execution" OFF)

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
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include"
                    "${PROJECT_SOURCE_DIR}/external_libs/include"
                    "${PROJECT_SOURCE_DIR}/tools/crfcodegen/include"
                    "${PROJECT_SOURCE_DIR}/tools/profiler/include"
                    "${ROCM_PATH}/include")
#link_directories("${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${ROCM_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")
link_directories("${ROCM_PATH}/hip/lib" "${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${ROCM_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")

#for test vectors default data path
add_definitions(-DTEST_VECTORS_DATA=\\"${PROJECT_SOURCE_DIR}/test_vectors/\\")
add_definitions(-DROCM_PATH=\\"${ROCM_PATH}/\\")

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
if(ROCM3)
	add_definitions(-DROCM3=1)
endif()

add_subdirectory(runtime)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(external_libs)
add_subdirectory(python_libs)
