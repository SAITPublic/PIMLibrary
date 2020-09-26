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

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hcc")
set(HCC_CXX_FLAGS
	"-hc --amdgpu-target=gfx900  --amdgpu-target=gfx906 --amdgpu-target=gfx908 ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_CXX_FLAGS}")

set(PY_FIM_INIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_fim_init.cpp)
set(PY_FIM_DEINIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_fim_deinit.cpp)
set(PY_FIM_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/py_fim_eltwise.cpp)

include_directories(/opt/rocm/include)

add_library(py_fim_init SHARED ${PY_FIM_INIT_SOURCES})
target_link_libraries(py_fim_init FimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_fim_init PROPERTY CXX_STANDARD 14)

add_library(py_fim_deinit SHARED ${PY_FIM_DEINIT_SOURCES})
target_link_libraries(py_fim_deinit FimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_fim_deinit PROPERTY CXX_STANDARD 14)

add_library(py_fim_eltwise SHARED ${PY_FIM_ELT_SOURCES})
target_link_libraries(py_fim_eltwise FimRuntime "${TORCH_LIBRARIES}")
set_property(TARGET py_fim_eltwise PROPERTY CXX_STANDARD 14)


install ( TARGETS py_fim_init LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS py_fim_deinit LIBRARY DESTINATION
           /opt/rocm/lib)
install ( TARGETS py_fim_eltwise LIBRARY DESTINATION
           /opt/rocm/lib)

install ( DIRECTORY python DESTINATION /opt/rocm/lib/py_fim_ops )
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION /opt/rocm/lib/py_fim_ops)
