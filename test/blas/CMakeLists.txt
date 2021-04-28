cmake_minimum_required(VERSION 3.1)

# Read project version
file(READ version.txt version_txt)

#
# Test BLAS project
#
project(TestBLAS
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "TestBLAS version " ${PROJECT_VERSION})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#
# Options
#

find_package(Catch2 REQUIRED
    HINTS ${CMAKE_SOURCE_DIR}/thirdparty)
if (Catch2_FOUND)
    message(STATUS "Catch2 found at " ${Catch2_DIR})
else()
    message(STATUS "Catch2 not found!")
endif()

# We ask for a `BLASConfig.cmake` on the C++-based BLAS
# to link diferent BLAS projects to TestBLAS.
#
find_package(BLAS)
if (BLAS_FOUND)
    message(STATUS "C++ BLAS found at " ${BLAS_DIR})
else()
    include_directories( "../../include" )
    message(STATUS "Using C++ BLAS from ${CMAKE_SOURCE_DIR}/include")
endif()

# # BLAS 'tblas.hpp' header file in C++
# find_path(blas_inc NAMES "tblas.hpp"
#     HINTS ${CMAKE_SOURCE_DIR}/include)
# if(blas_inc)
#     message(STATUS "tblas.hpp => " ${blas_inc})
#     include_directories( ${blas_inc} )
# else()
#     message(STATUS "tblas.hpp was NOT FOUND")
# endif(blas_inc)

if (OPENMP_FOUND)
    # set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#
# Targets
#

# C++ file for tests with corner cases
add_custom_target(corner-cases
    COMMAND
        cd "${CMAKE_CURRENT_SOURCE_DIR}/scripts" && ./wrapper_tests.py
        > "${CMAKE_CURRENT_SOURCE_DIR}/src/test_corner_cases.cpp"
    )

# Create tester 
set(tester "tester")
file(GLOB test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp"
    )
add_executable(tester tests_main.cpp ${test_sources})
target_include_directories(tester PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tester PRIVATE Catch2::Catch2)
if( USE_BLASPP_WRAPPERS )
    target_link_libraries(tester PRIVATE blaspp blas)
endif()

# Run tests
add_custom_target(run-tests
    COMMAND ./${tester}
    DEPENDS ${tester}
    )

#
# CTest
#

include(CTest)
include(Catch)
catch_discover_tests(tester)