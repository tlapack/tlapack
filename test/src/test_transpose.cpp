/// @file test_transpose.cpp
/// @brief Test transpose
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "testutils.hpp"
#include <tlapack.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Conjugate Transpose gives correct result", "[util]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Generate n
    idx_t n = GENERATE(1, 2, 3, 5, 10);
    // Generate m
    idx_t m = GENERATE(1, 2, 3, 5, 10);

    // Define the matrices
    std::vector<T> A_; auto A = new_matrix( A_, m, n );
    std::vector<T> B_; auto B = new_matrix( B_, n, m );

    // Generate a random matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    DYNAMIC_SECTION("Conjugate Transpose with"
                    << " m = " << m << " n = " << n)
    {
        transpose_opts_t<idx_t> opts;
        // Set nx to a small value so that the blocked algorithm gets tested even for small n and m;
        opts.nx = 3;
        conjtranspose(A, B, opts);

        for (idx_t i = 0; i < m; ++i)
            for (idx_t j = 0; j < n; ++j)
                CHECK(B(j, i) == conj(A(i, j)));
    }
}

TEMPLATE_LIST_TEST_CASE("Transpose gives correct result", "[util]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // Generate n
    idx_t n = GENERATE(1, 2, 3, 5, 10);
    // Generate m
    idx_t m = GENERATE(1, 2, 3, 5, 10);

    // Define the matrices
    std::vector<T> A_; auto A = new_matrix( A_, m, n );
    std::vector<T> B_; auto B = new_matrix( B_, n, m );

    // Generate a random matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    DYNAMIC_SECTION("Transpose with"
                    << " m = " << m << " n = " << n)
    {
        transpose_opts_t<idx_t> opts;
        // Set nx to a small value so that the blocked algorithm gets tested even for small n and m;
        opts.nx = 3;
        transpose(A, B, opts);

        for (idx_t i = 0; i < m; ++i)
            for (idx_t j = 0; j < n; ++j)
                CHECK(B(j, i) == A(i, j));
    }
}
