if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    install (DIRECTORY dramsim2  DESTINATION /opt/rocm/include)
endif()
