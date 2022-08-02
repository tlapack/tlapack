/// @file test_trtri.cpp
/// @brief Test TRTRI
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("TRTRI is stable", "[trtri]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    idx_t n = GENERATE(1, 2, 6, 9);

    const real_t eps = ulp<real_t>();
    const real_t tol = n * eps;

    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> C_(new T[n * n]);

    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto C = legacyMatrix<T, layout<matrix_t>>(n, n, &C_[0], n);

    // Generate random matrix in Schur form
    for (idx_t j = 0; j < n; ++j)
    {
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();
        A(j, j) += tlapack::make_scalar<T>(n, 0);
    }

    lacpy(uplo, A, C);

    DYNAMIC_SECTION("n = " << n << ", " << uplo)
    {
        trtri_recursive(uplo, C);

        // Calculate residuals

        if (uplo == Uplo::Lower)
        {
            for (idx_t j = 0; j < n; j++)
                for (idx_t i = 0; i < j; i++)
                    C(i, j) = T(0);
        }
        else
        {
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = 0; j < i; j++)
                    C(i, j) = T(0);
        }

        // TRMM with X starting as the inverse of C and leaving as the identity. This checks that the inverse is correct.
        // Note: it would be nice to have a ``upper * upper`` MM function to do this
        trmm(Side::Left, uplo, Op::NoTrans, Diag::NonUnit, T(1), A, C);

        for (idx_t i = 0; i < n; ++i)
            C(i, i) = C(i, i) - T(1);

        real_t normres = lantr(max_norm, uplo, Diag::NonUnit, C) / (lantr(max_norm, uplo, Diag::NonUnit, A));
        CHECK(normres <= tol);
    }
}