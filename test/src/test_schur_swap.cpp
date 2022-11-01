/// @file test_schur_swap.cpp
/// @brief Test 1x1 and 2x2 schur swaps
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack/plugins/debugutils.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("schur swap gives correct result", "[eigenvalues]", types_to_test)
{    
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    const T zero(0);
    const T one(1);
    idx_t n = 10;

    idx_t n1, n2, j;
    j = GENERATE(0, 1, 6);

    if (is_complex<T>::value)
    {
        n1 = 1;
        n2 = 1;
    }
    else
    {
        n1 = GENERATE(1, 2);
        n2 = GENERATE(1, 2);
    }
    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * n * eps;

    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[n * n]);
    std::unique_ptr<T[]> A_copy_(new T[n * n]);

    auto A = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &A_[0], n);
    auto Q = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &Q_[0], n);
    auto A_copy = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &A_copy_[0], n);

    // Generate random matrix in Schur form
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = zero;

    if( n1 == 2)
        A( j + 1, j ) = rand_helper<T>();
    if (n2 == 2)
        A(j + n1 + 1, j + n1) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);
    laset(Uplo::General, zero, one, Q);

    DYNAMIC_SECTION("j = " << j << " n1 = " << n1 << " n2 =" << n2)
    {
        schur_swap(true, A, Q, j, n1, n2);
        // Calculate residuals

        std::unique_ptr<T[]> _res(new T[n * n]);
        std::unique_ptr<T[]> _work(new T[n * n]);

        auto res = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &_res[0], n);
        auto work = legacyMatrix<T, idx_t, layout<matrix_t>>(n, n, &_work[0], n);
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::frob_norm, A);
        auto simil_res_norm = check_similarity_transform(A_copy, Q, A, res, work);
        CHECK(simil_res_norm <= tol * normA);

    }
}