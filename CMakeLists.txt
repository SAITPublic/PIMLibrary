cmake_minimum_required(VERSION 2.6)
project(FIM_LIBRARY)

set(FIM_LIBRARY_VERSION_MAJOR 1)
set(FIM_LIBRARY_VERSION_MINOR 0)

set(PREFIX ${CMAKE_INSTRALL_PREFIX})
set(EXEC_PREFIX "${PREFIX}/bin")
set(ROCM_PATH $ENV{ROCM_PATH})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BUILD_DIR})
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/hip/bin/hipcc")

include_directories("${PROJECT_SOURCE_DIR}/runtime/include" "${ROCM_PATH}/include")
link_directories("${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${PROJECT_SOURCE_DIR}/external_libs")

add_subdirectory(runtime/source)
add_subdirectory(examples)
#add_subdirectory(tools)
