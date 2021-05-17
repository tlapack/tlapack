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

set( BLAS_SIZE_T "c_int64_t" )
set( BLAS_INT_T  "c_int64_t" )
configure_file( cblas.fi.in ${PROJECT_SOURCE_DIR}/include/cblas.fi )
configure_file( blas.f90.in ${PROJECT_SOURCE_DIR}/include/blas.f90 )