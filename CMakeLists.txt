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
# T-LAPACK template library definitions
set( tlapack_defs "" )

#-------------------------------------------------------------------------------
# Options
option( USE_BLASPP_WRAPPERS   "Use BLAS++ wrappers"   OFF )
option( USE_BLASPP_TEMPLATES  "Use BLAS++ templates"  OFF )
option( USE_LAPACKPP_WRAPPERS "Use LAPACK++ wrappers" ON  )
option( BUILD_EXAMPLES        "Build examples"        ON  )
option( USE_GNU_MPFR          "Use GNU MPFR library"  OFF )

#-------------------------------------------------------------------------------
# Modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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
if(USE_BLASPP_WRAPPERS)
  list( APPEND tlapack_defs "-DUSE_BLASPP_WRAPPERS" )
endif()
if(USE_BLASPP_TEMPLATES)
  list( APPEND tlapack_defs "-DUSE_BLASPP_TEMPLATES" )
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
  list( APPEND tlapack_defs "-DUSE_LAPACKPP_WRAPPERS" )
endif()

#-------------------------------------------------------------------------------
# Search for MPFR library if it is needed
if( USE_GNU_MPFR )
  # find_package( GMP REQUIRED )
  find_package( MPFR REQUIRED )
  list( APPEND tlapack_defs "-DUSE_GNU_MPFR" )
endif()

#-------------------------------------------------------------------------------
# Options for BLAS_SIZE_T

if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES OR USE_LAPACKPP_WRAPPERS )
  set( BLAS_SIZE_T int64_t )
else()
  set( BLAS_SIZE_T size_t CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension" )
  set_property( CACHE BLAS_SIZE_T
    PROPERTY STRINGS
    intmax_t size_t 
    int8_t  uint8_t
    int16_t uint16_t
    int32_t uint32_t
    int64_t uint64_t )
endif()

if( NOT BLAS_SIZE_T STREQUAL "" )
  list( APPEND tlapack_defs "-DBLAS_SIZE_T=${BLAS_SIZE_T}" )
endif()

#-------------------------------------------------------------------------------
# Options for BLAS_INT_T

set( blas_intt_list 
  int
  short
  long
  "long long"
  int8_t
  int16_t
  int32_t
  int64_t
  int_least8_t
  int_least16_t
  int_least32_t
  int_least64_t
  int_fast8_t
  int_fast16_t
  int_fast32_t
  int_fast64_t
  intmax_t
  intptr_t
  ptrdiff_t )
  
if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES OR USE_LAPACKPP_WRAPPERS )
  set( BLAS_INT_T int64_t )
else()
  set( BLAS_INT_T int64_t CACHE STRING
    "Integer type for all variables except array sizes and matrix leading dimension" )
  set_property( CACHE BLAS_INT_T PROPERTY STRINGS ${blas_intt_list} )
endif()

if( NOT BLAS_INT_T STREQUAL "" )
  list( APPEND tlapack_defs "-DBLAS_INT_T=${BLAS_INT_T}" )
endif()

#-------------------------------------------------------------------------------
# Include directory for all targets
include_directories( "${PROJECT_SOURCE_DIR}/include" )

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