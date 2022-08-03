/// @file test_lauum.cpp
/// @brief Test LAUUM
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("LAUUM is stable", "[lauum]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    idx_t n = GENERATE(1, 2, 6, 9);

    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * n * eps;

    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> C_(new T[n * n]);

    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto C = legacyMatrix<T, layout<matrix_t>>(n, n, &C_[0], n);

    // Generate random matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, C);

    DYNAMIC_SECTION("n = " << n << " uplo = " << (uplo == Uplo::Upper ? "upper" : "lower"))
    {
        lauum_recursive(uplo, A);

        // Calculate residual
        real_t normC = lantr(max_norm, uplo, Diag::NonUnit, C);

        if (uplo == Uplo::Lower)
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    C(i, j) = T(0.);
            herk(Uplo::Lower, Op::ConjTrans, real_t(1), C, real_t(-1), A);
        }
        else
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j + 1; i < n; ++i)
                    C(i, j) = T(0.);
            herk(Uplo::Upper, Op::NoTrans, real_t(1), C, real_t(-1), A);
        }

        real_t res = lanhe(max_norm, uplo, A) / normC / normC;
        CHECK(res <= tol);
    }
}