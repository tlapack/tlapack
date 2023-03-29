/// @file test_gebd2.cpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @brief Test GEBD2 using UNG2R and UNGL2. Output an upper bidiagonal matrix B
/// for a m-by-n matrix A (m >= n).
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

    m = GENERATE(10, 15, 20, 30);
    n = GENERATE(10, 12, 20, 30);

    if (m >= n)  // Only m >= n matrices are supported (yet). gebd2 will give
                 // upper bidiagonal matrix B
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(max(m, n)) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, m, n);
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, m, m);
        std::vector<T> Z_;
        auto Z = new_matrix(Z_, n, n);
        auto Z11 = slice(Z, range(1, n), range(1, n));

        std::vector<T> tauv(n);  // min of m and n
        std::vector<T> tauw(n);  // min of m and n

        // Generate random m-by-n matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i, j) = rand_helper<T>();

        lacpy(Uplo::General, A, A_copy);

        DYNAMIC_SECTION("m = " << m << " n = " << n)
        {
            gebd2(A, tauv, tauw);

            // Get upper bidiagonal B
            std::vector<T> B_;
            auto B = new_matrix(B_, m, n);
            laset(Uplo::General, zero, zero, B);

            B(0, 0) = A(0, 0);
            for (idx_t j = 1; j < n; ++j) {
                B(j - 1, j) = A(j - 1, j);
                B(j, j) = A(j, j);
            }
            real_t normB = lange(Norm::Max, B);

            // Generate unitary matrix Q of m-by-m
            lacpy(Uplo::Lower, A, Q);

            ung2r(Q, tauv);

            // Test for Q's orthogonality
            std::vector<T> Wq_;
            auto Wq = new_matrix(Wq_, m, m);
            auto orth_Q = check_orthogonality(Q, Wq);
            CHECK(orth_Q / tol <= one);

            // Generate unitary matrix Z of n-by-n
            laset(Uplo::General, zero, one,
                  Z);  // Initialize Z as identity matrix.

            // Slice Z down to Z11 of size (n-1)-by-(n-1) and copy upper A to
            // Z11
            auto X = slice(A, range(0, n - 1),
                           range(1, n));  // X is (n-1)-by-(n-1) slice of A
            lacpy(Uplo::General, X, Z11);

            ungl2(Z11, tauw);  // Note: the unitary matrix Z we get
                               // here is ConjTransed

            // Test for Z's orthogonality
            std::vector<T> Wz_;
            auto Wz = new_matrix(Wz_, n, n);
            laset(Uplo::General, zero, one, Wz);
            auto orth_Z = check_orthogonality(Z, Wz);
            CHECK(orth_Z / tol <= one);

            // Test B = Q_H * A * Z
            // Generate a zero matrix K of size m-by-n to be the product of Q_H
            // * A
            std::vector<T> K_;
            auto K = new_matrix(K_, m, n);
            laset(Uplo::General, zero, zero, K);
            gemm(Op::ConjTrans, Op::NoTrans, real_t(1.), Q, A_copy, real_t(0),
                 K);

            // B = K * Z - B
            gemm(Op::NoTrans, Op::ConjTrans, real_t(1.), K, Z, real_t(-1.), B);

            real_t repres = lange(Norm::Max, B);
            CHECK(repres <= tol * normB);
        }
    }
}
