cmake_minimum_required(VERSION 3.1)

# Read project version:
file(READ version.txt version_txt)

project(TLAPACK
    VERSION ${version_txt}
    LANGUAGES CXX)

# Version:
message(STATUS "T-LAPACK version " ${PROJECT_VERSION})

#
# Options:
#

# option(BLASPP "Use BLASPP templates" ON)
# if(BLASPP)
#     find_path(blaspp_inc NAMES "blas.hh")
#     if(blaspp_inc)
#         message(STATUS "blas.hh => ${blaspp_inc}")
#         find_library(blaspp_lib "blaspp")
#         message(STATUS "libblaspp => ${blaspp_lib}")
#     else(blaspp_inc)
#         message(STATUS "blas.hh was NOT FOUND")
#     endif(blaspp_inc)
# endif(BLASPP)

# Options:
# option(OPTIMIZED_BLAS "Use an optimized BLAS library" ON)

# if(OPTIMIZED_BLAS)
#     find_library(blas_lib blas)
#     message(STATUS "libblas   => ${blas_lib}")
#     find_library(cblas_lib cblas)
#     message(STATUS "libcblas   => ${cblas_lib}")
# endif(OPTIMIZED_BLAS)

add_subdirectory(examples)
add_subdirectory(config)
add_subdirectory(test)