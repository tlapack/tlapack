/// @file test_hessenberg_rq.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test RQ reduction of Hessenberg matrix
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
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/hessenberg_rq.hpp>
#include <tlapack/lapack/rot_sequence.hpp>

#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

using namespace tlapack;

/** An unoptimized version of rot_sequence for testing purposes
 *
 *  @copybrief rot_sequence()
 *  @copydetails rot_sequence()
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_DIRECTION direction_t,
          TLAPACK_SVECTOR C_t,
          TLAPACK_SVECTOR S_t,
          TLAPACK_SMATRIX A_t>
int rot_sequence_unoptimized(
    side_t side, direction_t direction, const C_t& c, const S_t& s, A_t& A)
{
    using idx_t = size_type<A_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = (side == Side::Left) ? m - 1 : n - 1;

    // quick return
    if (k < 1) return 0;

    if (direction == Direction::Forward) {
        if (side == Side::Left) {
            for (idx_t i2 = k; i2 > 0; --i2) {
                idx_t i = i2 - 1;
                auto a1 = row(A, i);
                auto a2 = row(A, i + 1);
                rot(a1, a2, c[i], s[i]);
            }
        }
        else {
            for (idx_t i2 = k; i2 > 0; --i2) {
                idx_t i = i2 - 1;
                auto a1 = col(A, i);
                auto a2 = col(A, i + 1);
                rot(a1, a2, c[i], conj(s[i]));
            }
        }
    }
    else {
        if (side == Side::Left) {
            for (idx_t i = 0; i < k; ++i) {
                auto a1 = row(A, i);
                auto a2 = row(A, i + 1);
                rot(a1, a2, c[i], s[i]);
            }
        }
        else {
            for (idx_t i = 0; i < k; ++i) {
                auto a1 = col(A, i);
                auto a2 = col(A, i + 1);
                rot(a1, a2, c[i], conj(s[i]));
            }
        }
    }

    return 0;
}

TEMPLATE_TEST_CASE("RQ of Hessenberg matrix is accurate",
                   "[auxiliary]",
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
    PCG32 gen;

    const idx_t n = GENERATE(2, 3, 4, 5, 10, 13);

    const idx_t k = n - 1;

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(k + 1) * eps;

    // Define the matrices and vectors
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<real_t> cl(k);
    std::vector<T> sl(k);
    std::vector<real_t> cr(k);
    std::vector<T> sr(k);

    mm.random(H);
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = j + 1; i < n; ++i) {
            H(i, j) = (T)0;
        }
    }

    for (idx_t i = 0; i < k; ++i) {
        T t1 = rand_helper<T>(gen);
        T t2 = rand_helper<T>(gen);
        rotg(t1, t2, cl[i], sl[i]);
    }

    tlapack::lacpy(GENERAL, H, R);

    DYNAMIC_SECTION(" n = " << n)
    {
        hessenberg_rq(R, cl, sl, cr, sr);

        // Check backward error
        rot_sequence_unoptimized(LEFT_SIDE, FORWARD, cl, sl, H);
        rot_sequence_unoptimized(RIGHT_SIDE, FORWARD, cr, sr, H);
        real_t hnorm = lange(MAX_NORM, H);
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < n; ++i) {
                H(i, j) -= R(i, j);
            }
        }
        real_t res_norm = lange(MAX_NORM, H);
        CHECK(res_norm <= tol * hnorm);
    }
}
