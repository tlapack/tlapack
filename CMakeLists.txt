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
    if (OPENMP_FOUND)
        # set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

#
# Include subdirectories
#

add_subdirectory(examples)
add_subdirectory(config)
add_subdirectory(test)