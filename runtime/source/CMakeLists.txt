aux_source_directory(. runtime_source)
aux_source_directory(manager  runtime_source)
aux_source_directory(executor runtime_source)
aux_source_directory(emulator runtime_source)
aux_source_directory(utility  runtime_source)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HIPCC_FLAGS "--amdgpu-target=gfx906")
set(CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIPCC_FLAGS}")

if(DEFINED ENV{FIM_LOG_LEVEL})
    add_definitions(-DFIM_LOG_LEVEL=$ENV{FIM_LOG_LEVEL})
else()
    add_definitions(-DFIM_LOG_LEVEL=1)
endif()

message(STATUS "FIM Log level is " $ENV{FIM_LOG_LEVEL})

add_library(FimRuntime SHARED ${runtime_source})
target_link_libraries (FimRuntime hsakmt dramsim2 glog gflags)

install ( TARGETS FimRuntime LIBRARY DESTINATION ${ROCM_PATH}/lib/)
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/fim_runtime_api.h DESTINATION ${ROCM_PATH}/include )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/fim_data_types.h DESTINATION ${ROCM_PATH}/include )
