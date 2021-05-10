project(PY_CUST_OP)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(PYTHON_BIN_PATH "python")
set(ROCM_PATH $ENV{ROCM_PATH})

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Torch REQUIRED)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(HCC_CXX_FLAGS
	" --amdgpu-target=gfx900  --amdgpu-target=gfx906 --amdgpu-target=gfx908 ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

set(PY_PIM_INIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_init.cpp)
set(PY_PIM_DEINIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_deinit.cpp)
set(PY_PIM_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_eltwise.cpp)
set(PY_PIM_BN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_bn.cpp)
set(PY_PIM_ACTIVATION_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_activation.cpp)
set(PY_PIM_GEMV_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_gemv.cpp)
set(PY_PIM_LSTM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_lstm.cpp)
set(PY_PIM_DENSE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_pim_dense.cpp)

include_directories(${ROCM_PATH}/include)

add_library(py_pim_init SHARED ${PY_PIM_INIT_SOURCES})
target_link_libraries(py_pim_init PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_init PROPERTY CXX_STANDARD 14)

add_library(py_pim_deinit SHARED ${PY_PIM_DEINIT_SOURCES})
target_link_libraries(py_pim_deinit PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_deinit PROPERTY CXX_STANDARD 14)

add_library(py_pim_eltwise SHARED ${PY_PIM_ELT_SOURCES})
target_link_libraries(py_pim_eltwise PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_eltwise PROPERTY CXX_STANDARD 14)

add_library(py_pim_bn SHARED ${PY_PIM_BN_SOURCES})
target_link_libraries(py_pim_bn PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_bn PROPERTY CXX_STANDARD 14)

add_library(py_pim_activation SHARED ${PY_PIM_ACTIVATION_SOURCES})
target_link_libraries(py_pim_activation PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_activation PROPERTY CXX_STANDARD 14)

add_library(py_pim_gemv SHARED ${PY_PIM_GEMV_SOURCES})
target_link_libraries(py_pim_gemv PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_gemv PROPERTY CXX_STANDARD 14)

add_library(py_pim_lstm SHARED ${PY_PIM_LSTM_SOURCES})
target_link_libraries(py_pim_lstm MIOpen "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_lstm PROPERTY CXX_STANDARD 14)

add_library(py_pim_dense SHARED ${PY_PIM_DENSE_SOURCES})
target_link_libraries(py_pim_dense PimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_pim_dense PROPERTY CXX_STANDARD 14)

install ( TARGETS py_pim_init LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_deinit LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_eltwise LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_bn LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_activation LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_gemv LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_lstm LIBRARY DESTINATION
	${ROCM_PATH}/lib)
install ( TARGETS py_pim_dense LIBRARY DESTINATION
	           ${ROCM_PATH}/lib)
install ( DIRECTORY python DESTINATION ${ROCM_PATH}/lib/py_pim_ops )
install ( DIRECTORY modules DESTINATION ${ROCM_PATH}/lib/py_pim_ops )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${ROCM_PATH}/lib/py_pim_ops)
