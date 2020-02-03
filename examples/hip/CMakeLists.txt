add_executable(eltwise_add_hip eltwise_add_hip.cpp)
target_link_libraries(eltwise_add_hip FimRuntime)

add_executable(gemv_hip gemv_hip.cpp)
target_link_libraries(gemv_hip FimRuntime)
