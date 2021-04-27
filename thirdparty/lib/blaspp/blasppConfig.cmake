cmake_minimum_required( VERSION 3.15 )

set( blaspp_use_openmp "TRUE" )
set( blaspp_use_cuda   "FALSE" )
set( blaspp_use_hip    "FALSE" )

include( CMakeFindDependencyMacro )
if (blaspp_use_openmp)
    find_dependency( OpenMP )
endif()

# Export private variables used in LAPACK++.
set( blaspp_defines         "-DBLAS_FORTRAN_ADD_;-DBLAS_HAVE_CBLAS" )
set( blaspp_libraries       "-lblas;OpenMP::OpenMP_CXX" )

set( blaspp_cblas_found     "true" )
set( blaspp_cblas_include   "" )
set( blaspp_cblas_libraries "" )

include( "${CMAKE_CURRENT_LIST_DIR}/blasppTargets.cmake" )
