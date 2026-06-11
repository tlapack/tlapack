/// @file test_geqrt3.cpp
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
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/geqrt3.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/lansy.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/ung2r.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("geqrt3 computes the QR factorization of a matrix",
                   "[geqrt3][qrt]",
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

    m = GENERATE(5, 7, 63, 199, 512);
    n = GENERATE(2, 3, 5, 8, 16, 21, 51, 128);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(100 * n) * eps;
        real_t norm_orth;
        real_t normA;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> T_;
        auto Tmatrix = new_matrix(T_, n, n);
        std::vector<T> tau(std::min(m, n));

        // Generate a random matrix in A
        mm.random(A);
        mm.random(Tmatrix);

        // Check that the factorization was successful
        if (m <= 0 || n <= 0 || m < n) {
            norm_orth = real_t(0.0);
        }
        else {
            // Compute the QR factorization of A
            tlapack::geqrt3(A, Tmatrix);

            for (idx_t i = 0; i < n; ++i) {
                tau[i] = Tmatrix(i, i);
            }
            // Generates Q = H_1 H_2 ... H_n
            tlapack::ung2r(A, tau);

            // Compute ||QᴴQ - I||ᶠ
            std::vector<T> work_;
            auto work = new_matrix(work_, n, n);
            for (size_t j = 0; j < n; ++j)
                for (size_t i = 0; i < n; ++i)
                    work(i, j) = static_cast<float>(0xABADBABE);

            // work receives the identity n*n
            tlapack::laset(tlapack::UPPER_TRIANGLE, static_cast<T>(0.0),
                           static_cast<T>(1.0), work);
            // work receives QᴴQ - I
            tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                          static_cast<T>(1.0), A, A, static_cast<T>(-1.0),
                          work);

            // Compute ||QᴴQ - I||ᶠ
            norm_orth = tlapack::lansy(tlapack::FROB_NORM,
                                       tlapack::UPPER_TRIANGLE, work);
        }

        CHECK(norm_orth <= tol);
    }
}
