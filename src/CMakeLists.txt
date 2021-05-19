# # specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#-------------------------------------------------------------------------------
# Library: C wrappers for TBLAS
add_library(cwrappers_tblas cwrappers_tblas.cpp)

target_compile_definitions( cwrappers_tblas PUBLIC ${tlapack_defs} )
if( USE_BLASPP_WRAPPERS OR USE_BLASPP_TEMPLATES )
    target_link_libraries( cwrappers_tblas PUBLIC blaspp )
endif()
if( USE_GNU_MPFR )
    target_include_directories(cwrappers_tblas PRIVATE ${MPFR_INCLUDES} ${GMP_INCLUDES})
    target_link_libraries( cwrappers_tblas PRIVATE ${MPFR_LIBRARIES} ${GMP_LIBRARIES} )
endif()

set_target_properties( cwrappers_tblas
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

#-------------------------------------------------------------------------------
# Library: Fortran wrappers for TBLAS

# Integer types mapped to the ISO_FORTRAN_ENV module
if( (${BLAS_SIZE_T} IN_LIST blas_intt_list) OR
    (${BLAS_SIZE_T} STREQUAL "size_t") )
  string(REPLACE " " "_" CBLAS_SIZE_T_0 ${BLAS_SIZE_T})
  set( CBLAS_SIZE_T "c_${CBLAS_SIZE_T_0}" CACHE STRING
  "Integer type for array sizes, array indexes, and matrix leading dimension"
  FORCE )
else()
  string(REGEX REPLACE "^u" "" CBLAS_SIZE_T_0 ${BLAS_SIZE_T})
  if( ${CBLAS_SIZE_T_0} IN_LIST blas_intt_list )
    string(REPLACE " " "_" CBLAS_SIZE_T_0 ${CBLAS_SIZE_T_0})
    set( CBLAS_SIZE_T "c_${CBLAS_SIZE_T_0}" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension"
    FORCE )
  else()
    set( CBLAS_SIZE_T "" CACHE STRING
    "Integer type for array sizes, array indexes, and matrix leading dimension" )
  endif()
endif()

# Integer types mapped to the ISO_FORTRAN_ENV module
if( ${BLAS_INT_T} IN_LIST blas_intt_list )
  string(REPLACE " " "_" CBLAS_INT_T_0 ${BLAS_INT_T})
  set( CBLAS_INT_T "c_${CBLAS_INT_T_0}" CACHE STRING
  "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension"
  FORCE )
else()
  set( CBLAS_INT_T "" CACHE STRING
  "Integer type from ISO_FORTRAN_ENV for all variables except array sizes and matrix leading dimension" )
endif()

# Fortran interface. Used to call C-based BLAS directly from a Fortran program.
# The Fortran code needs to import this interface and link with the library cwrappers_tblas
configure_file( cblas.fi.in ${PROJECT_SOURCE_DIR}/include/cblas.fi @ONLY )

# Fortran library. This library can be linked directly to a Fortran code.
configure_file( fortranwrappers_tblas.f90.in ${CMAKE_CURRENT_SOURCE_DIR}/fortranwrappers_tblas.f90 @ONLY )

# Fortran module. Translates the C-based BLAS interface into Fortran 90.
# The Fortran code needs to load this interface and link with the library cwrappers_tblas
configure_file( blas.f90.in ${PROJECT_SOURCE_DIR}/include/blas.f90 @ONLY )

add_library(fortranwrappers_tblas fortranwrappers_tblas.f90)
target_link_libraries( fortranwrappers_tblas PUBLIC cwrappers_tblas )

set_target_properties( fortranwrappers_tblas
PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )