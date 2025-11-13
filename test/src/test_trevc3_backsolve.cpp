/// @file test_trevc_backsolve.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test eigenvector calculations.
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/gemv.hpp>
#include <tlapack/lapack/trevc.hpp>
#include <tlapack/lapack/trevc3_backsolve.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "TREVC3_backsolve correctly computes a block of right eigenvectors",
    "[eigenvalues][eigenvectors][trevc3]",
    TLAPACK_LEGACY_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;
    using range = pair<idx_t, idx_t>;
    using vector_t = vector_type<TestType>;

    // Eigenvalues and 16-bit types do not mix well
    if constexpr (sizeof(real_t) <= 2) SKIP_TEST;

    // Functor
    Create<matrix_t> new_matrix;
    Create<vector_t> new_vector;

    // MatrixMarket reader
    MatrixMarket mm;

    const int seed = GENERATE(2, 3);

    const idx_t n = GENERATE(1, 2, 3, 4, 5, 8, 10);
    const real_t zero(0);
    const real_t one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);

    mm.random(Uplo::Upper, T);
    // Set lower triangle to zero
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            T(i, j) = TA(zero);

    // Randomly set some subdiagonal entries to non-zero to create 2x2 blocks
    if constexpr (is_real<TA>) {
        idx_t j = 0;
        while (j + 1 < n) {
            if (rand_helper<float>(mm.gen) < 0.8f) {
                // Generate a 2x2 block in normalized form
                TA alpha = rand_helper<TA>(mm.gen);
                TA beta = rand_helper<TA>(mm.gen);
                TA gamma = rand_helper<TA>(mm.gen);
                T(j, j) = alpha;
                T(j, j + 1) = beta;
                T(j + 1, j) = -gamma;
                T(j + 1, j + 1) = alpha;
                j += 2;
            }
            else {
                j += 1;
            }
        }
    }

    // Calculate eigenvectors using trevc
    std::vector<TA> Vr_;
    auto Vr = new_matrix(Vr_, n, n);
    std::vector<TA> Vl_;
    auto Vl = new_matrix(Vl_, 0, 0);
    std::vector<TA> work_;
    auto work = new_vector(work_, n * 3);
    std::vector<real_t> rwork_;
    auto rwork = new_vector(rwork_, n);

    auto select = std::vector<bool>(n, true);
    trevc(Side::Right, HowMny::All, select, T, Vl, Vr, rwork, work);

    idx_t nb = 3;

    std::vector<real_t> colN_(n);
    auto colN = new_vector(colN_, n);
    for (idx_t j = 0; j < n; ++j) {
        idx_t itmax = iamax(slice(col(T, j), range(0, n)));
        colN[j] = abs1(T(itmax, j));
    }

    for (idx_t k = 0; k < n;) {
        idx_t nk = std::min(nb, n - k);
        idx_t ks = k;
        idx_t ke = k + nk;

        // Make sure we don't split 2x2 blocks
        if constexpr (is_real<TA>) {
            if (ke < n) {
                if (T(ke, ke - 1) != TA(zero)) {
                    ke += 1;
                    nk += 1;
                }
            }
        }

        DYNAMIC_SECTION(" n = " << n << " seed = " << seed << " ks = " << ks
                                << " ke = " << ke)
        {
            std::vector<TA> X_;
            auto X = new_matrix(X_, n, nk);

            std::vector<TA> work2_;
            auto work2 = new_vector(work2_, n * 3);

            // Compute the block of eigenvectors using trevc3_backsolve
            trevc3_backsolve(T, X, colN, work2, ks, ke, 4);

            // Compare the recomputed block with the original block
            real_t normDiff = zero;
            for (idx_t j = 0; j < nk; ++j) {
                for (idx_t i = 0; i < n; ++i) {
                    normDiff += abs(X(i, j) - Vr(i, ks + j));
                }
            }

            real_t Vrnorm =
                lange(Norm::Fro, slice(Vr, range(0, n), range(ks, ke)));

            real_t tol = ulp<real_t>() * real_t(n) * real_t(5);
            REQUIRE(normDiff <= tol * Vrnorm);

            // TODO: Maybe just test this in the same way as trevc_backsolve?
            // Comparing against Vr may cause issues if the eigenvectors are
            // ill-conditioned
        }

        k += nk;
    }
}