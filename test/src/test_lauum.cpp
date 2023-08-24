/// @file test_lauum.cpp
/// @author Heidi Meier, University of Colorado Denver
/// @brief Test LAUUM
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>

// Other routines
#include <tlapack/lapack/lantr.hpp>
#include <tlapack/lapack/lauum_recursive.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("LAUUM is stable", "[lauum]", TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    idx_t n = GENERATE(1, 2, 6, 9);

    const real_t eps = uroundoff<real_t>();
    const real_t tol = real_t(1.0e2 * n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, n, n);

    // Generate random matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(GENERAL, A, C);

    DYNAMIC_SECTION("n = " << n << " uplo = " << uplo)
    {
        lauum_recursive(uplo, A);

        // Calculate residual
        real_t normC = lantr(MAX_NORM, uplo, NON_UNIT_DIAG, C);

        if (uplo == Uplo::Lower) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    C(i, j) = T(0.);
            herk(LOWER_TRIANGLE, CONJ_TRANS, real_t(1), C, real_t(-1), A);
        }
        else {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j + 1; i < n; ++i)
                    C(i, j) = T(0.);
            herk(UPPER_TRIANGLE, NO_TRANS, real_t(1), C, real_t(-1), A);
        }

        real_t res = lanhe(MAX_NORM, uplo, A) / normC / normC;
        CHECK(res <= tol);
    }
}