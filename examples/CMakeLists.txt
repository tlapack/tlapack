# add the example version
add_executable(version version.cpp)
target_include_directories(version PUBLIC "${PROJECT_BINARY_DIR}/config")

# add the example gemm
add_executable(example_gemm example_gemm.cpp)
target_include_directories(example_gemm PUBLIC "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(example_gemm PUBLIC ${blaspp_lib} ${blas_lib} "-lgfortran")