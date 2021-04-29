# add the example version
add_executable(version version.cpp)
target_include_directories(version PRIVATE "${PROJECT_BINARY_DIR}/config")

# add the example gemm
add_executable(example_gemm
    example_gemm.cpp)
target_compile_definitions( example_gemm
    PRIVATE ${tlapack_defines} )
target_include_directories( example_gemm
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
target_link_libraries( example_gemm
    PRIVATE blaspp )

# add the example geqr
add_executable(example_geqr
    example_geqr.cpp)
target_compile_definitions( example_geqr
    PRIVATE ${tlapack_defines} )
target_include_directories( example_geqr
    PRIVATE "${PROJECT_SOURCE_DIR}/include" )
target_link_libraries( example_geqr
    PRIVATE lapackpp )