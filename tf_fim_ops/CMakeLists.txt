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

set(FIM_ACT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_activation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_activation_ops.cc)

set(FIM_BN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_bn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_bn_ops.cc)

set(FIM_LSTM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_lstm.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_lstm_ops.cc)

set(FIM_INIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_init.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_init_ops.cc)

set(FIM_DEINIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/fim_deinit.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/fim_deinit_ops.cc)

include_directories(/opt/rocm/include)

add_library(fim_gemv SHARED ${FIM_GEMV_SOURCES})
target_link_libraries(fim_gemv FimRuntime ${TF_LFLAGS})

add_library(fim_eltwise SHARED ${FIM_ELT_SOURCES})
target_link_libraries(fim_eltwise FimRuntime ${TF_LFLAGS})

add_library(fim_activation SHARED ${FIM_ACT_SOURCES})
target_link_libraries(fim_activation FimRuntime ${TF_LFLAGS})

add_library(fim_bn SHARED ${FIM_BN_SOURCES})
target_link_libraries(fim_bn FimRuntime ${TF_LFLAGS})

add_library(fim_lstm SHARED ${FIM_LSTM_SOURCES})
target_link_libraries(fim_lstm MIOpen ${TF_LFLAGS})

add_library(fim_init SHARED ${FIM_INIT_SOURCES})
target_link_libraries(fim_init FimRuntime ${TF_LFLAGS})

add_library(fim_deinit SHARED ${FIM_DEINIT_SOURCES})
target_link_libraries(fim_deinit FimRuntime ${TF_LFLAGS})

install ( TARGETS fim_gemv LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_eltwise LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_activation LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_bn LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_lstm LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_init LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS fim_deinit LIBRARY DESTINATION
           /opt/rocm/lib)

install ( DIRECTORY python DESTINATION /opt/rocm/lib/tf_fim_ops )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION /opt/rocm/lib/tf_fim_ops)
