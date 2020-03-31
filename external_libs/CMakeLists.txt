add_subdirectory(include)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
install ( FILES  libdramsim2.so DESTINATION /opt/rocm/lib )
endif()
