set(SOURCE0 fill_test.cpp)
#set(SOURCE1 multi_fill_test.cpp)
file(GLOB CRF_GEN_SOURCES ../../../../tools/crfcodegen/source/*.cpp)
cmake_minimum_required(VERSION 3.5.1)

include_directories(../../../../tools/crfcodegen/include)

add_executable(fill_test.out ${SOURCE0} ${CRF_GEN_SOURCES})
#add_executable(multi_fill_test.out ${SOURCE1} ${CRF_GEN_SOURCES})

target_link_libraries(fill_test.out)
#target_link_libraries(multi_fill_test.out hsakmt)

