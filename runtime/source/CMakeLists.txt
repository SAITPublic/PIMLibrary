aux_source_directory(. runtime_source)
aux_source_directory(manager  runtime_source)
aux_source_directory(executor runtime_source)
if(EMULATOR)
aux_source_directory(emulator runtime_source)
endif()
aux_source_directory(utility  runtime_source)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HIPCC_FLAGS "--amdgpu-target=gfx906 --amdgpu-target=gfx908")
set(CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIPCC_FLAGS}")

if(DEFINED ENV{PIM_LOG_LEVEL})
    add_definitions(-DPIM_LOG_LEVEL=$ENV{PIM_LOG_LEVEL})
else()
    add_definitions(-DPIM_LOG_LEVEL=1)
endif()

message(STATUS "PIM Log level is " $ENV{PIM_LOG_LEVEL})

add_library(PimRuntime SHARED ${runtime_source})

if(TARGET)
    target_link_libraries (PimRuntime hsakmt glog gflags)
else()
    target_link_libraries (PimRuntime hsakmt dramsim2 glog gflags)
endif()

install ( TARGETS PimRuntime LIBRARY DESTINATION ${ROCM_PATH}/lib/)
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/pim_runtime_api.h DESTINATION ${ROCM_PATH}/include )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/pim_data_types.h DESTINATION ${ROCM_PATH}/include )
