#-------------------------------------------------------------------------------
# Test sources
file( GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp" )

#-------------------------------------------------------------------------------
# Test library
add_library( testBLAS ${test_sources} )

target_compile_definitions( testBLAS PUBLIC ${tlapack_defs} )
target_include_directories( testBLAS
    PUBLIC "${PROJECT_SOURCE_DIR}/include" ${tblas_include_dir} )

if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES )
    target_link_libraries( testBLAS PUBLIC blaspp )
endif()
target_link_libraries( testBLAS PUBLIC Catch2::Catch2 )

set_target_properties( testBLAS
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
target_include_directories( testBLAS INTERFACE "${CMAKE_BINARY_DIR}/lib" )