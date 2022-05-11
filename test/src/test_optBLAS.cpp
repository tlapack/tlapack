/// @file test_optBLAS.cpp
/// @brief Test optBLAS wrappers from <T>LAPACK.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEST_CASE("has_compatible_layout gives the correct result", "[optBLAS]")
{
    using matrixA_t = legacyMatrix< float, Layout::ColMajor >;
    using matrixB_t = legacyMatrix< float, Layout::RowMajor >;
    using vector_t  = legacyVector< float >;

    CHECK( has_compatible_layout< matrixA_t, matrixA_t > );
    CHECK(!has_compatible_layout< matrixA_t, matrixB_t > );
    CHECK( has_compatible_layout< matrixB_t, matrixB_t > );
    CHECK(!has_compatible_layout< matrixB_t, matrixA_t > );

    CHECK( has_compatible_layout< matrixA_t, vector_t > );
    CHECK( has_compatible_layout< vector_t, matrixA_t > );
    CHECK( has_compatible_layout< matrixB_t, vector_t > );
    CHECK( has_compatible_layout< vector_t, matrixB_t > );

    CHECK( has_compatible_layout< matrixA_t, matrixA_t, matrixA_t > );
    CHECK(!has_compatible_layout< matrixA_t, matrixA_t, matrixB_t > );
    CHECK(!has_compatible_layout< matrixA_t, matrixB_t, matrixA_t > );
    CHECK(!has_compatible_layout< matrixA_t, matrixB_t, matrixB_t > );
    CHECK(!has_compatible_layout< matrixB_t, matrixA_t, matrixA_t > );
    CHECK(!has_compatible_layout< matrixB_t, matrixA_t, matrixB_t > );
    CHECK(!has_compatible_layout< matrixB_t, matrixB_t, matrixA_t > );
    CHECK( has_compatible_layout< matrixB_t, matrixB_t, matrixB_t > );
}

TEST_CASE("allow_optblas_v does not allow bool, int, long int, char", "[optBLAS]")
{
    CHECK(!allow_optblas_v<bool> );
    CHECK(!allow_optblas_v<int> );
    CHECK(!allow_optblas_v<long double> );
    CHECK(!allow_optblas_v<char> );
}

TEMPLATE_LIST_TEST_CASE("allow_optblas_v gives the correct result", "[optBLAS]", types_to_test)
{
    using matrix_t  = TestType;
    using alpha_t   = type_t< matrix_t >;
    using beta_t    = alpha_t;

    const bool allowOptBLAS = 
    #ifdef USE_BLASPP_WRAPPERS
        true;
    #else
        false;
    #endif

    CHECK( allow_optblas_v<matrix_t> == allowOptBLAS );
    CHECK( allow_optblas_v<alpha_t> == allowOptBLAS );
    CHECK( allow_optblas_v<beta_t> == allowOptBLAS );

    using T = alpha_t;
    CHECK( allow_optblas_v< pair< matrix_t, T > > == allowOptBLAS );
    CHECK( allow_optblas_v< pair< alpha_t, T > > == allowOptBLAS );
    CHECK( allow_optblas_v< pair< beta_t, T > > == allowOptBLAS );
    
    // gemm:
    using T = alpha_t;
    CHECK( allow_optblas_v<
        pair< matrix_t, T >,
        pair< matrix_t, T >,
        pair< matrix_t, T >,
        pair< alpha_t,  T >,
        pair< beta_t,   T >
    > == allowOptBLAS );
}