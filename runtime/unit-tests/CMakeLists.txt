file(GLOB SOURCES FIM_MemTraceCoalescerUnitTest.cpp)

include_directories("${PROJECT_SOURCE_DIR}/runtime/include")
add_executable(FIMRuntimeUnitTest ${SOURCES})

target_link_libraries(FIMRuntimeUnitTest gtest_main gtest pthread FimRuntime)
