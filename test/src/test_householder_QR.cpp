/// @file test_householder_QR.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
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
#include <tlapack/lapack/gen_householder_q.hpp>
#include <tlapack/lapack/householder_lq.hpp>
#include <tlapack/lapack/householder_q_mul.hpp>
#include <tlapack/lapack/householder_ql.hpp>
#include <tlapack/lapack/householder_qr.hpp>
#include <tlapack/lapack/householder_rq.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QR, RQ, QL, LQ factorization of a general m-by-n matrix",
                   "[qr][qrf]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // Generate test case
    using variant_t = pair<HouseholderQRVariant, idx_t>;
    const variant_t variant =
        GENERATE((variant_t(HouseholderQRVariant::Blocked, 1)),
                 (variant_t(HouseholderQRVariant::Blocked, 2)),
                 (variant_t(HouseholderQRVariant::Blocked, 4)),
                 (variant_t(HouseholderQRVariant::Blocked, 5)),
                 (variant_t(HouseholderQRVariant::Level2, 1)));
    const idx_t m = GENERATE(5, 10, 20, 30);
    const idx_t n = GENERATE(5, 10, 20, 30);
    const idx_t nv = GENERATE(5, 10, 20, 30);  // number of Householder vectors
                                               // to be used to generate Q

    // Variants for QR, QL, RQ, LQ
    const HouseholderQRVariant variant_qr = variant.first;
    const HouseholderQLVariant variant_ql =
        (variant.first == HouseholderQRVariant::Blocked)
            ? HouseholderQLVariant::Blocked
            : HouseholderQLVariant::Level2;
    const HouseholderLQVariant variant_lq =
        (variant.first == HouseholderQRVariant::Blocked)
            ? HouseholderLQVariant::Blocked
            : HouseholderLQVariant::Level2;
    const HouseholderRQVariant variant_rq =
        (variant.first == HouseholderQRVariant::Blocked)
            ? HouseholderRQVariant::Blocked
            : HouseholderRQVariant::Level2;

    // Constants
    const idx_t nb = variant.second;
    const idx_t k = min(m, n);
    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * max(m, n)) * eps;
    const real_t zero(0);

    // Matrices and vectors
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, m, n);
    auto& L = R;
    std::vector<T> tau(k);

    // Generate random test case
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = rand_helper<T>();

    // Copy A to A_copy
    lacpy(GENERAL, A, A_copy);
    // Compute norm of A
    const real_t anorm = lange(MAX_NORM, A_copy);

    if (nv <= m) {
        DYNAMIC_SECTION("QR with m = " << m << " n = " << n
                                       << " variant = " << (char)variant_qr
                                       << " nb = " << nb << " nv = " << nv)
        {
            // New matrices
            std::vector<T> Q_;
            auto Q = new_matrix(Q_, m, nv);

            // QR decomposition
            HouseholderQROpts qrOpts;
            qrOpts.variant = variant_qr;
            qrOpts.nb = nb;
            householder_qr(A, tau, qrOpts);

            // Copy A to Q and R
            lacpy(LOWER_TRIANGLE, slice(A, range(0, m), range(0, min(nv, k))),
                  Q);
            laset(LOWER_TRIANGLE, zero, zero, R);
            lacpy(UPPER_TRIANGLE, A, R);

            // Test Q is unitary
            gen_householder_q(FORWARD, COLUMNWISE_STORAGE, Q,
                              slice(tau, range(0, min(nv, k))),
                              GenHouseholderQOpts{(size_t)nb});
            auto orth_Q = check_orthogonality(Q);
            CHECK(orth_Q <= tol);

            // Test A == Q * R if nv >= k
            if (nv >= k) {
                auto V = slice(A, range(0, m), range(0, k));
                householder_q_mul(LEFT_SIDE, NO_TRANS, FORWARD,
                                  COLUMNWISE_STORAGE, V, tau, R,
                                  HouseholderQMulOpts{(size_t)nb});
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        R(i, j) = A_copy(i, j) - R(i, j);
                real_t repres = lange(MAX_NORM, R);
                CHECK(repres <= tol * anorm);
            }
        }

        DYNAMIC_SECTION("QL with m = " << m << " n = " << n
                                       << " variant = " << (char)variant_ql
                                       << " nb = " << nb << " nv = " << nv)
        {
            // Copy A_copy to A
            lacpy(GENERAL, A_copy, A);

            // New matrices
            std::vector<T> Q_;
            auto Q = new_matrix(Q_, m, nv);

            // QL decomposition
            HouseholderQLOpts qlOpts;
            qlOpts.variant = variant_ql;
            qlOpts.nb = nb;
            householder_ql(A, tau, qlOpts);

            // Copy A to Q and L
            if (m > n) {
                auto A1 = slice(A, range(m - n, m), range(0, n));
                auto L0 = slice(L, range(0, m - n), range(0, n));
                auto L1 = slice(L, range(m - n, m), range(0, n));
                laset(GENERAL, zero, zero, L0);
                laset(UPPER_TRIANGLE, zero, zero, L1);
                lacpy(LOWER_TRIANGLE, A1, L1);

                auto V0 = slice(A, range(0, m - min(nv, n)),
                                range(n - min(nv, n), n));
                auto V1 = slice(A, range(m - min(nv, n), m),
                                range(n - min(nv, n), n));
                auto Q0 = slice(Q, range(0, m - min(nv, n)),
                                range(nv - min(nv, n), nv));
                auto Q1 = slice(Q, range(m - min(nv, n), m),
                                range(nv - min(nv, n), nv));
                lacpy(GENERAL, V0, Q0);
                lacpy(UPPER_TRIANGLE, V1, Q1);
            }
            else {
                auto A0 = slice(A, range(0, m), range(0, n - m));
                auto A1 = slice(A, range(0, m), range(n - m, n));
                auto L0 = slice(L, range(0, m), range(0, n - m));
                auto L1 = slice(L, range(0, m), range(n - m, n));
                lacpy(GENERAL, A0, L0);
                laset(UPPER_TRIANGLE, zero, zero, L1);
                lacpy(LOWER_TRIANGLE, A1, L1);

                auto V0 = slice(A, range(0, m - nv), range(n - nv, n));
                auto V1 = slice(A, range(m - nv, m), range(n - nv, n));
                auto Q0 = slice(Q, range(0, m - nv), range(0, nv));
                auto Q1 = slice(Q, range(m - nv, m), range(0, nv));
                lacpy(GENERAL, V0, Q0);
                lacpy(UPPER_TRIANGLE, V1, Q1);
            }

            // Test Q is unitary
            gen_householder_q(BACKWARD, COLUMNWISE_STORAGE, Q,
                              slice(tau, range(k - min(nv, k), k)),
                              GenHouseholderQOpts{(size_t)nb});
            auto orth_Q = check_orthogonality(Q);
            CHECK(orth_Q <= tol);

            // Test A == Q * L if nv >= k
            if (nv >= k) {
                const auto V = slice(A, range(0, m),
                                     (m > n) ? range(0, n) : range(n - m, n));
                householder_q_mul(LEFT_SIDE, NO_TRANS, BACKWARD,
                                  COLUMNWISE_STORAGE, V, tau, L,
                                  HouseholderQMulOpts{(size_t)nb});
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        L(i, j) = A_copy(i, j) - L(i, j);
                real_t repres = lange(MAX_NORM, L);
                CHECK(repres <= tol * anorm);
            }
        }
    }

    if (nv <= n) {
        DYNAMIC_SECTION("LQ with m = " << m << " n = " << n
                                       << " variant = " << (char)variant_lq
                                       << " nb = " << nb << " nv = " << nv)
        {
            // Copy A_copy to A
            lacpy(GENERAL, A_copy, A);

            // New matrices
            std::vector<T> Q_;
            auto Q = new_matrix(Q_, nv, n);

            // LQ decomposition
            HouseholderLQOpts lqOpts;
            lqOpts.variant = variant_lq;
            lqOpts.nb = nb;
            householder_lq(A, tau, lqOpts);

            // Copy A to Q and L
            lacpy(UPPER_TRIANGLE, slice(A, range(0, min(nv, k)), range(0, n)),
                  Q);
            laset(UPPER_TRIANGLE, zero, zero, L);
            lacpy(LOWER_TRIANGLE, A, L);

            // Test Q is unitary
            gen_householder_q(FORWARD, ROWWISE_STORAGE, Q,
                              slice(tau, range(0, min(nv, k))),
                              GenHouseholderQOpts{(size_t)nb});
            auto orth_Q = check_orthogonality(Q);
            CHECK(orth_Q <= tol);

            // Test A == L * Q
            if (nv >= k) {
                const auto V = slice(A, range(0, k), range(0, n));
                householder_q_mul(RIGHT_SIDE, NO_TRANS, FORWARD,
                                  ROWWISE_STORAGE, V, tau, L,
                                  HouseholderQMulOpts{(size_t)nb});
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        L(i, j) = A_copy(i, j) - L(i, j);
                real_t repres = lange(MAX_NORM, L);
                CHECK(repres <= tol * anorm);
            }
        }

        DYNAMIC_SECTION("RQ with m = " << m << " n = " << n
                                       << " variant = " << (char)variant_rq
                                       << " nb = " << nb << " nv = " << nv)
        {
            // Copy A_copy to A
            lacpy(GENERAL, A_copy, A);

            // New matrices
            std::vector<T> Q_;
            auto Q = new_matrix(Q_, nv, n);

            // RQ decomposition
            HouseholderRQOpts rqOpts;
            rqOpts.variant = variant_rq;
            rqOpts.nb = nb;
            householder_rq(A, tau, rqOpts);

            // Copy A to Q and R
            if (n > m) {
                auto A1 = slice(A, range(0, m), range(n - m, n));
                auto R0 = slice(R, range(0, m), range(0, n - m));
                auto R1 = slice(R, range(0, m), range(n - m, n));
                laset(GENERAL, zero, zero, R0);
                laset(LOWER_TRIANGLE, zero, zero, R1);
                lacpy(UPPER_TRIANGLE, A1, R1);

                auto V0 = slice(A, range(m - min(nv, m), m),
                                range(0, n - min(nv, m)));
                auto V1 = slice(A, range(m - min(nv, m), m),
                                range(n - min(nv, m), n));
                auto Q0 = slice(Q, range(nv - min(nv, m), nv),
                                range(0, n - min(nv, m)));
                auto Q1 = slice(Q, range(nv - min(nv, m), nv),
                                range(n - min(nv, m), n));
                lacpy(GENERAL, V0, Q0);
                lacpy(LOWER_TRIANGLE, V1, Q1);
            }
            else {
                auto A0 = slice(A, range(0, m - n), range(0, n));
                auto A1 = slice(A, range(m - n, m), range(0, n));
                auto R0 = slice(R, range(0, m - n), range(0, n));
                auto R1 = slice(R, range(m - n, m), range(0, n));
                lacpy(GENERAL, A0, R0);
                laset(LOWER_TRIANGLE, zero, zero, R1);
                lacpy(UPPER_TRIANGLE, A1, R1);

                auto V0 = slice(A, range(m - nv, m), range(0, n - nv));
                auto V1 = slice(A, range(m - nv, m), range(n - nv, n));
                auto Q0 = slice(Q, range(0, nv), range(0, n - nv));
                auto Q1 = slice(Q, range(0, nv), range(n - nv, n));
                lacpy(GENERAL, V0, Q0);
                lacpy(LOWER_TRIANGLE, V1, Q1);
            }

            // Test Q is unitary
            gen_householder_q(BACKWARD, ROWWISE_STORAGE, Q,
                              slice(tau, range(k - min(nv, k), k)),
                              GenHouseholderQOpts{(size_t)nb});
            auto orth_Q = check_orthogonality(Q);
            CHECK(orth_Q <= tol);

            // Test A == R * Q if nv >= k
            if (nv >= k) {
                const auto V = slice(A, (n > m) ? range(0, m) : range(m - n, m),
                                     range(0, n));
                householder_q_mul(RIGHT_SIDE, NO_TRANS, BACKWARD,
                                  ROWWISE_STORAGE, V, tau, R,
                                  HouseholderQMulOpts{(size_t)nb});
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        R(i, j) = A_copy(i, j) - R(i, j);
                real_t repres = lange(MAX_NORM, R);
                CHECK(repres <= tol * anorm);
            }
        }
    }
}
