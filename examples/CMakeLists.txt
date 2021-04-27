# add the example version
add_executable(version version.cpp)
target_include_directories(version PUBLIC "${PROJECT_BINARY_DIR}/config")

# add the example gemm
add_executable(example_gemm example_gemm.cpp)
target_include_directories(example_gemm PUBLIC "${PROJECT_SOURCE_DIR}/include")
if( USE_BLASPP_WRAPPERS )
    target_link_libraries(example_gemm PRIVATE blaspp blas)
endif()

# add the example gemm
add_executable(example_geqr example_geqr.cpp)
target_include_directories(example_geqr PUBLIC "${PROJECT_SOURCE_DIR}/include")
if( USE_LAPACKPP_WRAPPERS )
    target_link_libraries(example_geqr PRIVATE lapackpp blas lapack)
endif()