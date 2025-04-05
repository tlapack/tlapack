/// @file test_rot_sequence.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test application of sequence of rotations
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

// Other routines
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

TEMPLATE_TEST_CASE("Application of rotation sequence is accurate",
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

    const Side side = GENERATE(Side::Left, Side::Right);
    const Direction direction =
        GENERATE(Direction::Forward, Direction::Backward);
    const idx_t n = GENERATE(1, 2, 3, 4, 5, 10, 13);
    const idx_t m = GENERATE(1, 2, 3, 4, 5, 10, 13);

    const idx_t k = (side == Side::Left) ? m - 1 : n - 1;

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(k) * eps;

    if (k < 1) SKIP_TEST;

    // Define the matrices and vectors
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> B_;
    auto B = new_matrix(B_, m, n);
    std::vector<real_t> c(k);
    std::vector<T> s(k);

    mm.random(A);

    for (idx_t i = 0; i < k; ++i) {
        T t1 = rand_helper<T>(gen);
        T t2 = rand_helper<T>(gen);
        rotg(t1, t2, c[i], s[i]);
    }
    tlapack::lacpy(GENERAL, A, B);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " side = " << side
                           << " direction = " << direction)
    {
        rot_sequence(side, direction, c, s, A);

        rot_sequence_unoptimized(side, direction, c, s, B);

        real_t bnorm = lange(MAX_NORM, B);
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < m; ++i) {
                B(i, j) -= A(i, j);
            }
        }
        real_t res_norm = lange(MAX_NORM, B);

        CHECK(res_norm <= tol * bnorm);
    }
}
