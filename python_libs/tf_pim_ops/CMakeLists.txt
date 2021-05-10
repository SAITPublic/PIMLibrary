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

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")

set(PIM_GEMV_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_gemv.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_gemv_ops.cc)

set(PIM_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_eltwise.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_eltwise_ops.cc)

set(PIM_ACT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_activation.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_activation_ops.cc)

set(PIM_BN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_bn.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_bn_ops.cc)

set(PIM_LSTM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_lstm.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_lstm_ops.cc)

set(PIM_DENSE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_dense.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_dense_ops.cc)

set(PIM_INIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_init.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_init_ops.cc)

set(PIM_DEINIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_deinit.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_deinit_ops.cc)

include_directories("${ROCM_PATH}/include")
include_directories(/usr/local/lib/python3.6/dist-packages/tensorflow/include)

add_library(pim_gemv SHARED ${PIM_GEMV_SOURCES})
target_link_libraries(pim_gemv PimRuntime ${TF_LFLAGS})

add_library(pim_eltwise SHARED ${PIM_ELT_SOURCES})
target_link_libraries(pim_eltwise PimRuntime ${TF_LFLAGS})

add_library(pim_activation SHARED ${PIM_ACT_SOURCES})
target_link_libraries(pim_activation PimRuntime ${TF_LFLAGS})

add_library(pim_bn SHARED ${PIM_BN_SOURCES})
target_link_libraries(pim_bn PimRuntime ${TF_LFLAGS})

add_library(pim_lstm SHARED ${PIM_LSTM_SOURCES})
target_link_libraries(pim_lstm MIOpen ${TF_LFLAGS})

add_library(pim_dense SHARED ${PIM_DENSE_SOURCES})
target_link_libraries(pim_dense PimRuntime ${TF_LFLAGS})

add_library(pim_init SHARED ${PIM_INIT_SOURCES})
target_link_libraries(pim_init PimRuntime ${TF_LFLAGS})

add_library(pim_deinit SHARED ${PIM_DEINIT_SOURCES})
target_link_libraries(pim_deinit PimRuntime ${TF_LFLAGS})

install ( TARGETS pim_gemv LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_eltwise LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_activation LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_bn LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_lstm LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_dense LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_init LIBRARY DESTINATION
           ${ROCM_PATH}/lib)
install ( TARGETS pim_deinit LIBRARY DESTINATION
           ${ROCM_PATH}/lib)

install ( DIRECTORY python DESTINATION ${ROCM_PATH}/lib/tf_pim_ops )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${ROCM_PATH}/lib/tf_pim_ops)
