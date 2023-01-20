/// @file test_trtri.cpp
/// @author Heidi Meier, University of Colorado Denver
/// @brief Test TRTRI
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>

// Other routines
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/trtri_recursive.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TRTRI is stable", "[trtri]", TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    Diag diag = GENERATE(Diag::Unit, Diag::NonUnit);
    idx_t n = GENERATE(1, 2, 6, 9);

    INFO("uplo = " << uplo);
    INFO("diag = " << diag);
    INFO("n = " << n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, n, n);

    // Generate random matrix in Schur form
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();

        if (diag == Diag::NonUnit)
            A(j, j) += real_t(n);
        else
            A(j, j) = real_t(1);
    }

    lacpy(uplo, A, C);

    {
        trtri_recursive(uplo, diag, C);

        // Calculate residuals

        if (uplo == Uplo::Lower) {
            for (idx_t j = 0; j < n; j++)
                for (idx_t i = 0; i < j; i++)
                    C(i, j) = T(0);
        }
        else {
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = 0; j < i; j++)
                    C(i, j) = T(0);
        }

        // TRMM with X starting as the inverse of C and leaving as the identity.
        // This checks that the inverse is correct. Note: it would be nice to
        // have a ``upper * upper`` MM function to do this
        trmm(Side::Left, uplo, Op::NoTrans, diag, T(1), A, C);

        for (idx_t i = 0; i < n; ++i)
            C(i, i) = C(i, i) - T(1);

        real_t normres = lantr(max_norm, uplo, Diag::NonUnit, C) /
                         (lantr(max_norm, uplo, diag, A));
        CHECK(normres <= tol);
    }
}