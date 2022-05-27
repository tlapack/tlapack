/// @file test_transpose.cpp
/// @brief Test transpose
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Conjugate Transpose gives correct result", "[util]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Generate n
    idx_t n = GENERATE(1, 2, 3, 5, 10);
    // Generate m
    idx_t m = GENERATE(1, 2, 3, 5, 10);

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * m]);
    std::unique_ptr<T[]> B_(new T[n * m]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto B = legacyMatrix<T, layout<matrix_t>>(n, m, &B_[0], layout<matrix_t> == Layout::ColMajor ? n : m);

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

TEMPLATE_LIST_TEST_CASE("Conjugate Transpose gives correct result", "[util]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Generate n
    idx_t n = GENERATE(1, 2, 3, 5, 10);
    // Generate m
    idx_t m = GENERATE(1, 2, 3, 5, 10);

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * m]);
    std::unique_ptr<T[]> B_(new T[n * m]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n);
    auto B = legacyMatrix<T, layout<matrix_t>>(n, m, &B_[0], layout<matrix_t> == Layout::ColMajor ? n : m);

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
