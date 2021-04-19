cmake_minimum_required(VERSION 3.1)

project(TLAPACK
    VERSION 0.1.1
    LANGUAGES "CXX" "C" "Fortran")

# Version:
file(WRITE "version.txt" ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_VERSION})
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

# Options:
option(BLASPP "Use BLASPP templates" ON)
option(OPTIMIZED_BLAS "Use an optimized BLAS library" ON)
 
if(BLASPP)
    find_library(blaspp_lib blaspp)
    message(STATUS "libblaspp => ${blaspp_lib}")
    find_path(blaspp_inc
        NAMES "blas.hh")
    message(STATUS "blas.hh => ${blaspp_inc}")
endif(BLASPP)

if(OPTIMIZED_BLAS)
    find_library(blas_lib blas)
    message(STATUS "libblas   => ${blas_lib}")
    find_library(cblas_lib cblas)
    message(STATUS "libcblas   => ${cblas_lib}")
endif(OPTIMIZED_BLAS)

add_subdirectory(examples)
add_subdirectory(config)