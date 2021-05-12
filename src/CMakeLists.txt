# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#-------------------------------------------------------------------------------
# Library: C wrappers for TBLAS
add_library(cwrappers_tblas cwrappers_tblas.cpp)

target_compile_definitions( cwrappers_tblas PUBLIC ${tlapack_defs} )
target_include_directories( cwrappers_tblas PUBLIC "${PROJECT_SOURCE_DIR}/include" )
if( USE_BLASPP_WRAPPERS )
    target_link_libraries( cwrappers_tblas PUBLIC blaspp )
endif()

set_target_properties( cwrappers_tblas
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
target_include_directories( cwrappers_tblas INTERFACE "${CMAKE_BINARY_DIR}/lib" )