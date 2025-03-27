/// @file test_rot_sequence3.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test application of sequence of rotations
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
#include <tlapack/lapack/rot_sequence.hpp>
#include <tlapack/lapack/rot_sequence3.hpp>

#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("Application of rotation sequence is accurate",
                   "[auxiliary]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using real_matrix_t = real_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<real_matrix_t> new_real_matrix;

    // MatrixMarket reader
    MatrixMarket mm;
    rand_generator gen;

    const Side side = GENERATE(Side::Left, Side::Right);
    const Direction direction =
        GENERATE(Direction::Forward, Direction::Backward);
    const idx_t n = GENERATE(1, 2, 3, 4, 5, 10, 13);
    const idx_t m = GENERATE(1, 2, 3, 4, 5, 10, 13);
    const idx_t l = GENERATE(1, 2, 3, 4);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " l = " << l << " side = "
                           << side << " direction = " << direction)
    {
        const idx_t k = (side == Side::Left) ? m - 1 : n - 1;

        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(k) * eps;

        if (k < 1) SKIP_TEST;

        // Define the matrices and vectors
        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);
        std::vector<real_t> c_;
        auto C = new_real_matrix(c_, k, l);
        std::vector<T> s_;
        auto S = new_matrix(s_, k, l);

        mm.random(A);

        // Generate random rotation matrices
        for (idx_t j = 0; j < l; ++j) {
            for (idx_t i = 0; i < k; ++i) {
                T t1 = rand_helper<T>(gen);
                T t2 = rand_helper<T>(gen);
                rotg(t1, t2, C(i, j), S(i, j));
            }
        }
        tlapack::lacpy(GENERAL, A, B);

        // Apply the rotations
        rot_sequence3(side, direction, C, S, A);

        // Apply the rotations using rot_sequence
        for (idx_t j = 0; j < l; ++j) {
            auto c = col(C, j);
            auto s = col(S, j);
            rot_sequence(side, direction, c, s, B);
        }

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
