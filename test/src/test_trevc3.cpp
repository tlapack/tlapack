/// @file test_trevc3.cpp
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
#include <tlapack/lapack/trevc3.hpp>
#include <tlapack/lapack/trevc3_backsolve.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TREVC3_backsolve correctly computes the right eigenvector",
                   "[eigenvalues][eigenvectors][trevc3]",
                   TLAPACK_TYPES_TO_TEST)
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

    const idx_t n = GENERATE(1, 2, 3, 5, 8, 10);
    const real_t zero(0);
    const real_t one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);

    std::vector<TA> v_;
    auto v = new_vector(v_, n);

    mm.random(Uplo::Upper, T);
    // Set lower triangle to zero
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            T(i, j) = TA(zero);

    // Randomly set some subdiagonal entries to non-zero to create 2x2 blocks
    if constexpr (is_real<TA>) {
        // idx_t j = 0;
        // while (j + 1 < n) {
        //     if (rand_helper<float>(mm.gen) < 0.3f) {
        //         T(j + 1, j) = rand_helper<real_t>(mm.gen);
        //         j += 2;
        //     }
        //     else {
        //         j += 1;
        //     }
        // }

        // Set a 2x2 block for testing
        if (n >= 4) {
            T(1, 0) = TA(0.5);
        }
    }

    for (idx_t k = 0; k < n; ++k) {
        DYNAMIC_SECTION(" n = " << n << " seed = " << seed << " k = " << k)
        {
            if (k > 0) {
                if (T(k, k - 1) != TA(zero)) {
                    // Skip the second value of a 2x2 block
                    continue;
                }
            }

            bool is_2x2_block = false;
            if (k + 1 < n) {
                if (T(k + 1, k) != TA(zero)) {
                    is_2x2_block = true;
                }
            }

            if (is_2x2_block) {
                continue;
            }
            //
            // Compute right eigenvector using trevc3_backsolve_single
            //
            trevc3_backsolve_single(T, v, k);

            //
            // Verify that T*v = lambda*v
            //
            TA lambda = T(k, k);
            std::vector<TA> Tv_;
            auto Tv = new_vector(Tv_, n);
            gemv(Op::NoTrans, one, T, v, zero, Tv);

            real_t normv = asum(v);
            real_t tol = ulp<real_t>() * normv * real_t(n);
            for (idx_t i = 0; i < n; ++i) {
                CHECK(std::abs(Tv[i] - lambda * v[i]) <= tol);
            }
        }
    }
}
