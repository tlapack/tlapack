# add the example version
add_executable(version version.cpp)
target_include_directories(version PRIVATE "${PROJECT_BINARY_DIR}/config")

# add the example gemm
add_executable(example_gemm
    example_gemm.cpp)
target_compile_definitions( example_gemm
    PRIVATE ${tlapack_defs} )
target_include_directories( example_gemm
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
if( USE_BLASPP_WRAPPERS )
    target_link_libraries( example_gemm
        PRIVATE blaspp )
endif()

# add the example gemm using the C wrapper
add_executable(example_cwrapper_gemm
    example_cwrapper_gemm.c)
target_compile_definitions( example_cwrapper_gemm
    PRIVATE ${tlapack_defs} )
target_include_directories( example_cwrapper_gemm
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
target_link_libraries( example_cwrapper_gemm
    PRIVATE cwrappers_tblas )

# add the example saxpy using the Fortran module
add_executable( example_fortranModule_saxpy
    example_fortranModule_saxpy.f90 "${PROJECT_SOURCE_DIR}/include/blas.f90" )
target_link_libraries( example_fortranModule_saxpy
    PRIVATE cwrappers_tblas )

# add the example saxpy using the Fortran wrapper
add_executable( example_fortranWrapper_daxpy
    example_fortranWrapper_daxpy.f90 )
target_include_directories( example_fortranWrapper_daxpy
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
target_link_libraries( example_fortranWrapper_daxpy
    PRIVATE cwrappers_tblas )

# add the example geqr
add_executable(example_geqr
    example_geqr.cpp)
target_compile_definitions( example_geqr
    PRIVATE ${tlapack_defs} )
target_include_directories( example_geqr
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
if( USE_LAPACKPP_WRAPPERS )
    target_link_libraries( example_geqr
        PRIVATE lapackpp )
endif()