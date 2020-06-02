project(CUST_OP)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(PYTHON_BIN_PATH "python3")
set(ROCM_PATH $ENV{ROCM_PATH})

execute_process(COMMAND python3 -c
                "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')"
                 OUTPUT_VARIABLE TF_LFLAGS)
execute_process(COMMAND python3 -c
                "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')"
                OUTPUT_VARIABLE TF_CFLAGS)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hcc")
set(HCC_CXX_FLAGS
    "-hc --amdgpu-target=gfx900  --amdgpu-target=gfx906 --amdgpu-target=gfx908  ${TF_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

set(FIM_GEMV_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_gemv.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_gemv_ops.cc)

set(FIM_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_eltwise.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_eltwise_ops.cc)

set(MIOPEN_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/miopen_eltwise.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/miopen_eltwise_ops.cc)

set(MIOPEN_ACT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/miopen_activation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/miopen_activation_ops.cc)

set(MIOPEN_BN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/miopen_bn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/miopen_bn_ops.cc)
include_directories(/opt/rocm/include)

add_library(fim_gemv SHARED ${FIM_GEMV_SOURCES})
target_link_libraries(fim_gemv FimRuntime ${TF_LFLAGS})

add_library(fim_eltwise SHARED ${FIM_ELT_SOURCES})
target_link_libraries(fim_eltwise FimRuntime ${TF_LFLAGS})

add_library(miopen_eltwise SHARED ${MIOPEN_ELT_SOURCES})
target_link_libraries(miopen_eltwise MIOpen ${TF_LFLAGS})

add_library(miopen_activation SHARED ${MIOPEN_ACT_SOURCES})
target_link_libraries(miopen_activation MIOpen ${TF_LFLAGS})

add_library(miopen_bn SHARED ${MIOPEN_BN_SOURCES})
target_link_libraries(miopen_bn MIOpen ${TF_LFLAGS})

install ( TARGETS fim_gemv LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_eltwise LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS miopen_eltwise LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS miopen_activation LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS miopen_bn LIBRARY DESTINATION
           /opt/rocm/lib)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
install ( DIRECTORY python DESTINATION /opt/rocm/lib/tf_fim_ops )
endif()
