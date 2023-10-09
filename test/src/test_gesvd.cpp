/// @file test_gesvd.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test GESVD
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
#include <tlapack/lapack/laset.hpp>

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/gesvd.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("svd with small unitary matrix is backward stable",
                   "[svd]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);

    idx_t m, n;

    m = GENERATE(1, 4, 5, 10, 15);
    n = GENERATE(1, 4, 5, 10, 12);
    idx_t k = min(m, n);

    const int seed = GENERATE(2, 3, 4, 5, 6, 7, 8, 9, 10);
    rand_generator gen;
    gen.seed(seed);

    const real_t eps = ulp<real_t>();
    real_t tol = real_t(20. * n) * eps;
    // Use a slightly larger tolerance for half precision
    if (eps > real_t(1.0e-6)) tol = tol * real_t(5.);

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> U_;
    auto U = new_matrix(U_, m, k);
    std::vector<T> Vt_;
    auto Vt = new_matrix(Vt_, k, n);

    std::vector<real_t> s(k);

    // Generate random m-by-n matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>(gen);

    lacpy(Uplo::General, A, A_copy);
    real_t normA = lange(Norm::Max, A);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " seed = " << seed)
    {
        int err = gesvd(true, true, A, s, U, Vt);
        CHECK(err == 0);

        // Check that singular values are positive and sorted in decreasing
        // order
        for (idx_t i = 0; i < k; ++i) {
            CHECK(s[i] >= real_t(0));
        }
        for (idx_t i = 0; i + 1 < k; ++i) {
            CHECK(s[i] >= s[i + 1]);
        }

        // Test for U's orthogonality
        std::vector<T> Wu_;
        auto Wu = new_matrix(Wu_, k, k);
        auto orth_U = check_orthogonality(U, Wu);
        CHECK(orth_U <= tol);

        // Test for Vt's orthogonality
        std::vector<T> Wvt_;
        auto Wvt = new_matrix(Wvt_, k, k);
        auto orth_Vt = check_orthogonality(Vt, Wvt);
        CHECK(orth_Vt <= tol);

        // Get diagonal B
        std::vector<T> B_;
        auto B = new_matrix(B_, k, k);
        laset(Uplo::General, zero, zero, B);
        for (idx_t j = 0; j < k; ++j) {
            B(j, j) = s[j];
        }

        // Test U * B * V^H = A
        std::vector<T> K_;
        auto K = new_matrix(K_, m, k);
        laset(Uplo::General, zero, zero, K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), U, B, real_t(0), K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), K, Vt, real_t(-1.), A_copy);
        real_t repres = lange(Norm::Max, A_copy);
        CHECK(repres <= tol * normA);
    }
}

TEMPLATE_TEST_CASE("svd with full unitary matrix is backward stable",
                   "[svd]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);

    idx_t m, n;

    m = GENERATE(1, 4, 5, 10, 15);
    n = GENERATE(1, 4, 5, 10, 12);
    idx_t k = min(m, n);

    const int seed = GENERATE(2, 3, 4, 5, 6, 7, 8, 9, 10);
    rand_generator gen;
    gen.seed(seed);

    const real_t eps = ulp<real_t>();
    real_t tol = real_t(20. * n) * eps;
    // Use a slightly larger tolerance for half precision
    if (eps > real_t(1.0e-6)) tol = tol * real_t(5.);

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> U_;
    auto U = new_matrix(U_, m, m);
    std::vector<T> Vt_;
    auto Vt = new_matrix(Vt_, n, n);

    std::vector<real_t> s(k);

    // Generate random m-by-n matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>(gen);

    lacpy(Uplo::General, A, A_copy);
    real_t normA = lange(Norm::Max, A);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " seed = " << seed)
    {
        int err = gesvd(true, true, A, s, U, Vt);
        REQUIRE(err == 0);

        // Check that singular values are positive and sorted in decreasing
        // order
        for (idx_t i = 0; i < k; ++i) {
            CHECK(s[i] >= real_t(0));
        }
        for (idx_t i = 0; i + 1 < k; ++i) {
            CHECK(s[i] >= s[i + 1]);
        }

        // Test for U's orthogonality
        std::vector<T> Wu_;
        auto Wu = new_matrix(Wu_, m, m);
        auto orth_U = check_orthogonality(U, Wu);
        CHECK(orth_U <= tol);

        // Test for Vt's orthogonality
        std::vector<T> Wvt_;
        auto Wvt = new_matrix(Wvt_, n, n);
        auto orth_Vt = check_orthogonality(Vt, Wvt);
        CHECK(orth_Vt <= tol);

        // Get diagonal B
        std::vector<T> B_;
        auto B = new_matrix(B_, m, n);
        laset(Uplo::General, zero, zero, B);
        for (idx_t j = 0; j < k; ++j) {
            B(j, j) = s[j];
        }

        // Test U * B * V^H = A
        std::vector<T> K_;
        auto K = new_matrix(K_, m, n);
        laset(Uplo::General, zero, zero, K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), U, B, real_t(0), K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), K, Vt, real_t(-1.), A_copy);
        real_t repres = lange(Norm::Max, A_copy);
        CHECK(repres <= tol * normA);
    }
}