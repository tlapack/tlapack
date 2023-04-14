/// @file test_mul_rot_sequence.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test routine to efficiently apply a sequence of rotations
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
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/rotg.hpp>
#include <tlapack/lapack/mul_rot_sequence.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Apply sequence of rotations",
                   "[rot]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = real_type<T>;
    using real_matrix_t = legacyMatrix<real_t, std::size_t, Layout::ColMajor>;

    // Functor
    Create<real_matrix_t> new_real_matrix;
    Create<matrix_t> new_matrix;

    const T zero(0);

    idx_t m, n, nr, k;

    m = GENERATE(5, 10, 20);
    n = GENERATE(5, 10, 20);
    nr = GENERATE(1, 2, 5);
    auto side = GENERATE(Side::Left, Side::Right);
    k = side == Side::Left ? m : n;

    if (k < nr) return;

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<real_t> C_;
    auto C = new_real_matrix(C_, k - 1, nr);
    std::vector<T> S_;
    auto S = new_matrix(S_, k - 1, nr);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<real_t>();

    lacpy(Uplo::General, A, A_copy);

    for (idx_t j = 0; j < nr; ++j) {
        for (idx_t i = 0; i < k - 1; ++i) {
            T f = rand_helper<real_t>();
            T g = rand_helper<real_t>();
            rotg(f, g, C(i, j), S(i, j));
        }
    }

    DYNAMIC_SECTION(" m = " << m << " n = " << n << " nr = " << nr
                            << " side = " << side)
    {
        // Call the routine being tested
        mul_rot_sequence(side, C, S, A);

        // Manually do the rotations
        if (side == Side::Left) {
            for (idx_t j = 0; j < nr; ++j) {
                for (idx_t i = 0; i < k - 1; ++i) {
                    auto a1 = row(A_copy, i);
                    auto a2 = row(A_copy, i + 1);
                    rot(a1, a2, C(i, j), S(i, j));
                }
            }
        }
        else {
            for (idx_t j = 0; j < nr; ++j) {
                for (idx_t i = 0; i < k - 1; ++i) {
                    auto a1 = col(A_copy, i);
                    auto a2 = col(A_copy, i + 1);
                    rot(a1, a2, C(i, j), S(i, j));
                }
            }
        }

        real_t anorm = lange(Norm::Max, A_copy);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A_copy(i, j) -= A(i, j);
        real_t enorm = lange(Norm::Max, A_copy);
        CHECK(enorm <= tol * anorm);
    }
}
