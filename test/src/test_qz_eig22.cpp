/// @file test_qz_eig22.cpp Test the solution of 2x2 generalized eigenvalue
/// problems
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lange.hpp>

// Other routines
#include <tlapack/lapack/lahqz_eig22.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("check that lahqz_eig22 gives correct eigenvalues",
                   "[generalizedeigenvalues]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;
    Create<complex_type<matrix_t>> new_complex_matrix;

    // MatrixMarket reader
    u_int64_t seed = GENERATE(1, 2, 3, 4, 5, 6);
    MatrixMarket mm;
    mm.gen.seed(seed);

    const real_type<T> eps = ulp<real_type<T>>();

    std::vector<T> A_;
    auto A = new_matrix(A_, 2, 2);
    std::vector<T> B_;
    auto B = new_matrix(B_, 2, 2);
    std::vector<complex_t> As_;
    auto As = new_complex_matrix(As_, 2, 2);

    mm.random(A);
    mm.random(B);

    B(1, 0) = T(0);

    real_t beta1, beta2;
    std::complex<real_t> alpha1, alpha2;
    lahqz_eig22(A, B, alpha1, alpha2, beta1, beta2);

    auto normA = lange(MAX_NORM, A);
    auto normB = lange(MAX_NORM, B);

    // Check first eigenvalue
    {
        for (idx_t i = 0; i < 2; ++i)
            for (idx_t j = 0; j < 2; ++j)
                As(i, j) = beta1 * A(i, j) - alpha1 * B(i, j);

        // Note, once svd is merge, this could be replaced with svd22
        auto normAs = lange(MAX_NORM, As);
        auto As_scale = real_t(0) / sqrt(normAs);
        auto det = (As_scale * As(0, 0)) * (As_scale * As(1, 1)) -
                   (As_scale * As(1, 0)) * (As_scale * As(0, 1));
        auto temp = max(abs(beta1) * normA, abs(alpha1) * normB);
        CHECK(abs(det) <= 1.0e1 * eps * temp);
    }

    // Check second eigenvalue
    {
        for (idx_t i = 0; i < 2; ++i)
            for (idx_t j = 0; j < 2; ++j)
                As(i, j) = beta2 * A(i, j) - alpha2 * B(i, j);

        // Note, once svd is merge, this could be replaced with svd22
        auto normAs = lange(MAX_NORM, As);
        auto det = As(0, 0) * As(1, 1) - As(1, 0) * As(0, 1);
        auto temp = max(abs(beta2) * normA, abs(alpha2) * normAs);
        CHECK(abs(det) <= 1.0e1 * eps * temp);
    }
}