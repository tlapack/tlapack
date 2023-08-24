/// @file test_optBLAS.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test optBLAS wrappers
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("has_compatible_layout gives the correct result",
                   "[optBLAS]",
                   TLAPACK_TYPES_TO_TEST)
{
    using T = type_t<TestType>;
    using matrixA_t = TestType;
    using matrixB_t = decltype(transpose_view(std::declval<const matrixA_t>()));
    using vector_t = vector_type<TestType>;
    using namespace tlapack::traits::internal;

    CHECK(has_compatible_layout<T, T>);
    CHECK(has_compatible_layout<T, T, T>);
    CHECK(has_compatible_layout<matrixA_t, T>);
    CHECK(has_compatible_layout<T, matrixA_t>);
    CHECK(has_compatible_layout<matrixB_t, T>);
    CHECK(has_compatible_layout<T, matrixB_t>);
    CHECK(has_compatible_layout<vector_t, T>);
    CHECK(has_compatible_layout<T, vector_t>);

    CHECK(has_compatible_layout<matrixA_t, matrixA_t>);
    CHECK(!has_compatible_layout<matrixA_t, matrixB_t>);
    CHECK(has_compatible_layout<matrixB_t, matrixB_t>);
    CHECK(!has_compatible_layout<matrixB_t, matrixA_t>);

    CHECK(layout<vector_t> == Layout::Strided);
    CHECK(layout<matrixA_t> != layout<matrixB_t>);
    CHECK(has_compatible_layout<matrixA_t, vector_t>);
    CHECK(has_compatible_layout<vector_t, matrixA_t>);
    CHECK(has_compatible_layout<matrixB_t, vector_t>);
    CHECK(has_compatible_layout<vector_t, matrixB_t>);

    CHECK(has_compatible_layout<matrixA_t, matrixA_t, matrixA_t>);
    CHECK(!has_compatible_layout<matrixA_t, matrixA_t, matrixB_t>);
    CHECK(!has_compatible_layout<matrixA_t, matrixB_t, matrixA_t>);
    CHECK(!has_compatible_layout<matrixA_t, matrixB_t, matrixB_t>);
    CHECK(!has_compatible_layout<matrixB_t, matrixA_t, matrixA_t>);
    CHECK(!has_compatible_layout<matrixB_t, matrixA_t, matrixB_t>);
    CHECK(!has_compatible_layout<matrixB_t, matrixB_t, matrixA_t>);
    CHECK(has_compatible_layout<matrixB_t, matrixB_t, matrixB_t>);
}

TEST_CASE("allow_optblas does not allow bool, int, long int, char", "[optBLAS]")
{
    CHECK(!allow_optblas<bool>);
    CHECK(!allow_optblas<int>);
    CHECK(!allow_optblas<long double>);
    CHECK(!allow_optblas<char>);
}

// TEMPLATE_TEST_CASE("legacy_matrix() exists", "[utils]",
// TLAPACK_TYPES_TO_TEST)
// {
//     using matrix_t = TestType;
//     using legacy_t = decltype( legacy_matrix( std::declval<matrix_t>() ) );

//     CHECK ( is_same_v< matrix_t, legacy_t > );
// }

TEMPLATE_TEST_CASE("allow_optblas gives the correct result",
                   "[optBLAS]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;

    // Test matrix_t and pair< matrix_t, T >
    CHECK(allow_optblas<matrix_t> == allow_optblas<T>);
    CHECK(allow_optblas<pair<matrix_t, T>> == allow_optblas<T>);
    CHECK(allow_optblas<pair<T, T>> == allow_optblas<T>);

    // Test pairs of convertible types
    CHECK(allow_optblas<pair<float, double>> == allow_optblas<double>);
    CHECK(allow_optblas<pair<int, float>> == allow_optblas<float>);
    CHECK(allow_optblas<pair<double, std::complex<double>>> ==
          allow_optblas<std::complex<double>>);
    CHECK(allow_optblas<pair<double, long double>> ==
          allow_optblas<long double>);
    CHECK(allow_optblas<pair<std::complex<double>, double>> == false);

    // Test pair< matrix_t, T > and pairs of convertible types
    CHECK(allow_optblas<pair<matrix_t, T>, pair<T, T>> == allow_optblas<T>);
    CHECK(allow_optblas<pair<matrix_t, T>, pair<int, T>> == allow_optblas<T>);
    CHECK(allow_optblas<pair<matrix_t, T>, pair<float, T>> == allow_optblas<T>);
    CHECK(allow_optblas<pair<matrix_t, T>, pair<long double, T>> ==
          allow_optblas<T>);

    // Test pair< matrix_t, T > and pair< matrix_t, T >
    CHECK(allow_optblas<pair<matrix_t, T>, pair<matrix_t, T>> ==
          allow_optblas<T>);
}