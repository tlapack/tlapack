cmake_minimum_required(VERSION 3.1)

# Read project version
file(READ version.txt version_txt)

#
# T-LAPACK project
#
project(TLAPACK
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#
# Options
#

if(CMAKE_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DUSE_OPENMP)
    if(OPENMP_FOUND)
        # set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
        set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
    endif()
endif()

option( USE_BLASPP_WRAPPERS  "Use BLAS++ wrappers"  ON  )
option( USE_BLASPP_TEMPLATES "Use BLAS++ templates" OFF )
if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES )
    find_package(blaspp REQUIRED
        HINTS ${CMAKE_SOURCE_DIR}/thirdparty)
    if(blaspp_FOUND)
        message(STATUS "BLAS++ found at " ${blaspp_DIR})
        if(USE_BLASPP_WRAPPERS)
            set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BLASPP_WRAPPERS" )
        endif()
        if(USE_BLASPP_TEMPLATES)
            set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_BLASPP_TEMPLATES" )
        endif()
    else()
        message(STATUS "BLAS++ not found!")
    endif()
endif()

option( USE_LAPACKPP_WRAPPERS  "Use LAPACK++ wrappers"  ON  )
if( USE_LAPACKPP_WRAPPERS )
    find_package(lapackpp REQUIRED
        HINTS ${CMAKE_SOURCE_DIR}/thirdparty)
    if(lapackpp_FOUND)
        message(STATUS "LAPACK++ found at " ${lapackpp_DIR})
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LAPACKPP_WRAPPERS" )
    else()
        message(STATUS "LAPACK++ not found!")
    endif()
endif()

#
# Include subdirectories
#

add_subdirectory(docs)
add_subdirectory(config)
add_subdirectory(examples)
add_subdirectory(test)