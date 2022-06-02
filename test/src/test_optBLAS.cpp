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

#include <complex>

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

TEMPLATE_LIST_TEST_CASE("legacy_matrix() exists", "[utils]", types_to_test)
{
    using matrix_t = TestType;
    using legacy_t = decltype( legacy_matrix( std::declval<matrix_t>() ) );

    CHECK ( is_same_v< matrix_t, legacy_t > );
}

TEMPLATE_LIST_TEST_CASE("allow_optblas_v gives the correct result", "[optBLAS]", types_to_test)
{
    using matrix_t  = TestType;
    using T = type_t<matrix_t>;

    // Test matrix_t and pair< matrix_t, T >
    CHECK( allow_optblas_v<matrix_t> == allow_optblas_v<T> );
    CHECK( allow_optblas_v< pair< matrix_t, T > > == allow_optblas_v<T> );

    // Test pairs of convertible types
    CHECK( allow_optblas_v< pair< float, double > > == allow_optblas_v<double> );
    CHECK( allow_optblas_v< pair< int, float > > == allow_optblas_v<float> );
    CHECK( allow_optblas_v< pair< double, std::complex<double> > > == allow_optblas_v<std::complex<double>> );
    CHECK( allow_optblas_v< pair< double, long double > > == allow_optblas_v<long double> );
    CHECK( allow_optblas_v< pair< std::complex<double>, double > > == false );
    
    // Test pair< matrix_t, T > and pairs of convertible types
    CHECK( allow_optblas_v< pair< matrix_t, T >, pair< T, T > > == allow_optblas_v<T> );
    CHECK( allow_optblas_v< pair< matrix_t, T >, pair< int, T > > == allow_optblas_v<T> );
    CHECK( allow_optblas_v< pair< matrix_t, T >, pair< float, T > > == allow_optblas_v<T> );
    CHECK( allow_optblas_v< pair< matrix_t, T >, pair< long double, T > > == allow_optblas_v<T> );
    
    // Test pair< matrix_t, T > and pair< matrix_t, T >
    CHECK( allow_optblas_v< pair< matrix_t, T >, pair< matrix_t, T > > == allow_optblas_v<T> );
}