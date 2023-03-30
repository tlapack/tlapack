/// @file test_gebd2.cpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test GEBD2 using UNG2R and UNGL2. Output an upper/lower bidiagonal
/// matrix B for a m-by-n matrix A.
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
#include <tlapack/lapack/laset.hpp>
// #include <tlapack/plugins/debugutils.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/gebd2.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/lapack/ungl2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("bidiagonal reduction is backward stable",
                   "[bidiagonal][svd]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);
    const real_t one(1);

    idx_t m, n;

    m = GENERATE(1, 4, 5, 10, 15);
    n = GENERATE(1, 4, 5, 10, 12);
    idx_t k = min(m, n);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * max(m, n)) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, m);
    std::vector<T> Z_;
    auto Z = new_matrix(Z_, n, n);

    std::vector<T> tauv(k);
    std::vector<T> tauw(k);

    // Generate random m-by-n matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    lacpy(Uplo::General, A, A_copy);
    real_t normA = lange(Norm::Max, A);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        gebd2(A, tauv, tauw);

        // Get bidiagonal B
        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);
        laset(Uplo::General, zero, zero, B);

        if (m >= n) {
            // copy upper bidiagonal matrix
            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < n; ++j) {
                B(j - 1, j) = A(j - 1, j);
                B(j, j) = A(j, j);
            }
        }
        else {
            // copy lower bidiagonal matrix
            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < m; ++j) {
                B(j, j - 1) = A(j, j - 1);
                B(j, j) = A(j, j);
            }
        }

        // Generate m-by-m unitary matrix Q
        if (m >= n) {
            lacpy(Uplo::Lower, A, Q);
            ung2r(Q, tauv);
        }
        else {
            auto Q11 = slice(Q, range(1, m), range(1, m));
            auto X = slice(A, range(1, m), range(0, m - 1));
            auto tauv1 = slice(tauv, range(0, k - 1));
            laset(Uplo::General, zero, one, Q);
            lacpy(Uplo::Lower, X, Q11);
            ung2r(Q11, tauv1);
        }

        // Test for Q's orthogonality
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, m, m);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q / tol <= one);

        // Generate m-by-m unitary matrix Z
        if (m >= n) {
            auto Z11 = slice(Z, range(1, n), range(1, n));
            auto X = slice(A, range(0, n - 1), range(1, n));
            auto tauw1 = slice(tauw, range(0, k - 1));
            laset(Uplo::General, zero, one, Z);
            lacpy(Uplo::Upper, X, Z11);
            ungl2(Z11, tauw1);
        }
        else {
            lacpy(Uplo::Upper, A, Z);
            ungl2(Z, tauw);
        }

        // Test for Z's orthogonality
        std::vector<T> Wz_;
        auto Wz = new_matrix(Wz_, n, n);
        auto orth_Z = check_orthogonality(Z, Wz);
        CHECK(orth_Z / tol <= one);

        // Test Q * B * Z^H = A
        std::vector<T> K_;
        auto K = new_matrix(K_, m, n);
        laset(Uplo::General, zero, zero, K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), Q, B, real_t(0), K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), K, Z, real_t(-1.), A_copy);
        real_t repres = lange(Norm::Max, A_copy);
        CHECK(repres <= tol * normA);
    }
}
