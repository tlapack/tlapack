/// @file test/src/test_norms.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/lantr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Blue's constants work when computing norms",
                   "[norm]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    // constants
    const real_t u = uroundoff<real_t>();
    const real_t tbig = blue_max<real_t>();

    // Generators
    const idx_t n = GENERATE(2, pow(2, digits<real_t>() / 2));

    // Skip test if n is too large
    if (n > 10000) SKIP_TEST;

    // Create matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);

    // Tolerance
    const real_t tol = u;

    DYNAMIC_SECTION("lange, n = " << n)
    {
        // constants
        const real_t norm = tbig * real_t(n);

        mm.single_value(A, tbig);
        CHECK(abs(lange(FROB_NORM, A) - norm) <= tol * norm);
    }

    DYNAMIC_SECTION("lanhe, n = " << n)
    {
        // constants
        const real_t norm = tbig * real_t(n);

        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            INFO("uplo = " << uplo);
            mm.single_value(uplo, A, tbig);
            CHECK(abs(lanhe(FROB_NORM, uplo, A) - norm) <= tol * norm);
        }
    }

    DYNAMIC_SECTION("lansy, n = " << n)
    {
        // constants
        const real_t norm = tbig * real_t(n);

        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            INFO("uplo = " << uplo);
            mm.single_value(uplo, A, tbig);
            CHECK(abs(lansy(FROB_NORM, uplo, A) - norm) <= tol * norm);
        }
    }

    DYNAMIC_SECTION("lantr, n = " << n)
    {
        real_t norm;

        norm = tbig * sqrt(real_t((n * (n + 1)) / 2));
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            INFO("diag = NonUnit, uplo = " << uplo);
            mm.single_value(uplo, A, tbig);
            CHECK(abs(lantr(FROB_NORM, uplo, NON_UNIT_DIAG, A) - norm) <=
                  tol * norm);
        }

        norm = sqrt(tbig * tbig * real_t((n * (n - 1)) / 2) + real_t(n));
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            INFO("diag = Unit, uplo = " << uplo);
            mm.single_value(uplo, A, tbig);
            CHECK(abs(lantr(FROB_NORM, uplo, UNIT_DIAG, A) - norm) <=
                  tol * norm);
        }
    }
}