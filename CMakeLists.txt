cmake_minimum_required(VERSION 3.18)
# VERSION 3.18: set_source_files_properties( PROPERTIES Fortran_PREPROCESS ON )

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# T-LAPACK project
project(TLAPACK
    VERSION ${version_txt}
    LANGUAGES CXX C Fortran)
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# Options
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers"   OFF )
option( USE_BLASPP_TEMPLATES  "Use BLAS++ templates"  OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers" ON  )
option( BUILD_EXAMPLES        "Build examples"        ON  )

#-------------------------------------------------------------------------------
# Search for BLAS++ library if it is needed
if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES )
    find_package( blaspp )
    if(blaspp_FOUND)
        message( STATUS "  BLAS++ found in ${blaspp_DIR}" )
    else()
        message( FATAL_ERROR "  BLAS++ not found." )
    endif()
endif()

#-------------------------------------------------------------------------------
# Search for LAPACK++ library if it is needed
if( USE_LAPACKPP_WRAPPERS )
    find_package( lapackpp )
    if(lapackpp_FOUND)
        message( STATUS "  LAPACK++ found in ${lapackpp_DIR}" )
    else()
        message( FATAL_ERROR "  LAPACK++ not found." )
    endif()
endif()

set( tlapack_defs "" )

#-------------------------------------------------------------------------------
# Use BLAS++ wrappers
if(USE_BLASPP_WRAPPERS)
    list( APPEND tlapack_defs "-DUSE_BLASPP_WRAPPERS" )
endif()

#-------------------------------------------------------------------------------
# Use BLAS++ templates
if(USE_BLASPP_TEMPLATES)
    list( APPEND tlapack_defs "-DUSE_BLASPP_TEMPLATES" )
endif()

#-------------------------------------------------------------------------------
# Use LAPACK++ wrappers
if(USE_LAPACKPP_WRAPPERS)
    list( APPEND tlapack_defs "-DUSE_LAPACKPP_WRAPPERS" )
endif()

#-------------------------------------------------------------------------------
# Docs
add_subdirectory(docs)

#-------------------------------------------------------------------------------
# Config
add_subdirectory(config)

#-------------------------------------------------------------------------------
# C and Fortran wrappers
add_subdirectory(src)

#-------------------------------------------------------------------------------
# Examples
if( BUILD_EXAMPLES )
    add_subdirectory(examples)
endif()

#-------------------------------------------------------------------------------
# Include tests
include(CTest)
if( BUILD_TESTING )
    add_subdirectory(test)
endif()