aux_source_directory(. pimbench_source)
aux_source_directory(source pimbench_source)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
include_directories(./include)
add_executable(pimbench ${pimbench_source})
target_link_libraries(pimbench PimRuntime glog gflags Boost::program_options)
