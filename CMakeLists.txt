cmake_minimum_required(VERSION 3.1)

#-------------------------------------------------------------------------------
# Read project version
file(READ version.txt version_txt)

#-------------------------------------------------------------------------------
# T-LAPACK project
project(TLAPACK
    VERSION ${version_txt}
    LANGUAGES CXX)
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

#-------------------------------------------------------------------------------
# Options
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers"   ON  )
option( USE_BLASPP_TEMPLATES  "Use BLAS++ templates"  OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers" ON  )

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

set( tlapack_defines "" )

#-------------------------------------------------------------------------------
# Use BLAS++ wrappers
if(USE_BLASPP_WRAPPERS)
    list( APPEND tlapack_defines "-DUSE_BLASPP_WRAPPERS" )
endif()

#-------------------------------------------------------------------------------
# Use BLAS++ templates
if(USE_BLASPP_TEMPLATES)
    list( APPEND tlapack_defines "-DUSE_BLASPP_TEMPLATES" )
endif()

#-------------------------------------------------------------------------------
# Use LAPACK++ wrappers
if(USE_LAPACKPP_WRAPPERS)
    list( APPEND tlapack_defines "-DUSE_LAPACKPP_WRAPPERS" )
endif()

#-------------------------------------------------------------------------------
# Include subdirectories
add_subdirectory(docs)
add_subdirectory(config)
add_subdirectory(examples)

#-------------------------------------------------------------------------------
# Include tests
include(CTest)
if( BUILD_TESTING )
    add_subdirectory(test)
endif()