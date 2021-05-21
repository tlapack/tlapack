cmake_minimum_required(VERSION 3.1)

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# Test BLAS project
project(TestBLAS
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "TestBLAS version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# TODO: Define a `TBLASConfig.cmake` on the C++-based BLAS
set( tblas_include_dir "${CMAKE_SOURCE_DIR}/include" )
message( STATUS "tblas_include_dir = ${tblas_include_dir}" )
message( STATUS "tlapack_defs = ${tlapack_defs}" )

#-------------------------------------------------------------------------------
# Search for Catch2
find_package( Catch2 REQUIRED )
include(Catch)

#-------------------------------------------------------------------------------
# corner-cases: Generate C++ file for tests with corner cases
add_custom_target( corner-cases
COMMAND
    cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" && ./wrapper_tests.py
    > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp" )

#-------------------------------------------------------------------------------
# Test sources
file( GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp" )

# #-------------------------------------------------------------------------------
# # Test library
# add_library( testBLAS ${test_sources} )

# target_compile_definitions( testBLAS PUBLIC ${tlapack_defs} )
# target_include_directories( testBLAS
#     PUBLIC "${PROJECT_SOURCE_DIR}/include" ${tblas_include_dir} )

# if( USE_BLASPP )
#     target_link_libraries( testBLAS PUBLIC blaspp )
# endif()
# target_link_libraries( testBLAS PUBLIC Catch2::Catch2 )

# set_target_properties( testBLAS
#     PROPERTIES
#     ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#     LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

#-------------------------------------------------------------------------------
# tester: Program for tests

# # add_subdirectory( src )
# add_executable( tester tests_main.cpp )
# target_link_libraries( tester PRIVATE testBLAS )

add_executable( tester tests_main.cpp ${test_sources} )
target_compile_definitions( tester
    PRIVATE ${tlapack_defs} )
target_include_directories( tester
PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${tblas_include_dir} )
target_link_libraries( tester PRIVATE Catch2::Catch2 )
if( USE_BLASPP )
    target_link_libraries( tester PRIVATE blaspp )
endif()
if( USE_GNU_MPFR )
    target_include_directories(tester PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES})
    target_link_libraries( tester PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

set_target_properties( tester
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )

#-------------------------------------------------------------------------------
# Add tests to CTest
catch_discover_tests(tester)