/// @file test_geqr2.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/geqr2.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/lansy.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/ung2r.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("geqr2 computes the QR factorization of a matrix",
                   "[geqr2]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t m, n;

    m = GENERATE(16);
    n = GENERATE(16);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(100 * n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> tau(std::min(m, n));

        // Generate a random matrix in A
        mm.random(A);

        // Compute the norm of A
        auto normA = lange(FROB_NORM, A);

        // Check that the factorization was successful
        // if (m <= 0 || n <= 0 || m < n) {
        //     SKIP("m <= 0 || n <= 0 || m < n");
        // }

        // Compute the QR factorization of A
        geqr2(A, tau);

        // Generates Q = H_1 H_2 ... H_n
        ung2r(A, tau);

        // Compute ||Q'Q - I||_F
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                work(i, j) = T(static_cast<float>(0xABADBABE));

        // work receives the identity n*n
        laset(UPPER_TRIANGLE, static_cast<T>(0.0), static_cast<T>(1.0), work);
        // work receives Q'Q - I
        gemm(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A, A,
             static_cast<T>(-1.0), work);

        // Compute ||Q'Q - I||_F
        real_t norm_orth_1 = lansy(FROB_NORM, UPPER_TRIANGLE, work);

        CHECK((norm_orth_1 / normA) <= tol);
    }
}
