/// @file test_trevc.cpp
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
#include <tlapack/lapack/hessenberg.hpp>
#include <tlapack/lapack/lahqr.hpp>
#include <tlapack/lapack/trevc.hpp>
#include <tlapack/lapack/unghr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("TREVC correctly computes the eigenvectors",
                   "[eigenvalues][eigenvectors][trevc]",
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

    const idx_t n = GENERATE(1, 2, 3, 4, 5, 8, 10);
    const real_t zero(0);
    const real_t one(1);
    // TODO: add test for howmny = HowMny::Select
    const HowMny howmny = GENERATE(HowMny::All, HowMny::Back);
    const Side side = GENERATE(Side::Right, Side::Left, Side::Both);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<TA> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);
    std::vector<TA> tau_;
    auto tau = new_vector(tau_, n - 1);
    std::vector<complex_t> w_;
    auto w = new_vector(w_, n);

    mm.random(A);

    DYNAMIC_SECTION(" n = " << n << " seed = " << seed << " howmny = "
                            << (howmny == HowMny::All ? "All" : "Back")
                            << " side = "
                            << (side == Side::Right
                                    ? "Right"
                                    : (side == Side::Left ? "Left" : "Both")))
    {
        // Calculate the Schur decomposition A = Q*T*Q**T
        lacpy(Uplo::General, A, T);
        // Step 1: Reduce to Hessenberg form H = Q1**T * A * Q1
        hessenberg((idx_t)0, (idx_t)n, T, tau);
        // Step 2: Compute Q from the Householder vectors
        lacpy(Uplo::Lower, T, Q);
        unghr((idx_t)0, (idx_t)n, Q, tau);
        // Step 3: Reduce Hessenberg to real Schur form T = Q2**T * H * Q2
        // and update Q = Q1 * Q2
        // (set lower triangle of T to zero first)
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                T(i, j) = TA(zero);
        lahqr(true, true, (idx_t)0, (idx_t)n, T, w, Q);

        // Sanity check, make sure that A = Q*T*Q**T
        // Can be removed later as this just tests whether the test is
        // constructed correctly
        std::vector<TA> temp_;
        auto temp = new_matrix(temp_, n, n);
        std::vector<TA> A_reconstructed_;
        auto A_reconstructed = new_matrix(A_reconstructed_, n, n);
        gemm(Op::NoTrans, Op::NoTrans, one, Q, T, zero, temp);
        gemm(Op::NoTrans, Op::ConjTrans, one, temp, Q, zero, A_reconstructed);
        real_t normA = lange(Norm::One, A);
        for (idx_t i = 0; i < n; ++i)
            for (idx_t j = 0; j < n; ++j)
                A_reconstructed(i, j) -= A(i, j);
        real_t normDiff = lange(Norm::One, A_reconstructed);
        real_t tol = ulp<real_t>() * normA * real_t(n) * real_t(n) * real_t(5);
        REQUIRE(normDiff <= tol);

        // Now compute the eigenvectors using trevc
        bool calcLeft = (side == Side::Left || side == Side::Both);
        bool calcRight = (side == Side::Right || side == Side::Both);
        std::vector<TA> Vr_;
        auto Vr = calcRight ? new_matrix(Vr_, n, n) : new_matrix(Vr_, 0, 0);
        std::vector<TA> Vl_;
        auto Vl = calcLeft ? new_matrix(Vl_, n, n) : new_matrix(Vl_, 0, 0);
        std::vector<TA> work_;
        auto work = new_vector(work_, n * 3);

        auto select =
            std::vector<bool>(0, false);  // Not used for howmny != Select

        if (calcLeft and howmny == HowMny::Back) lacpy(Uplo::General, Q, Vl);
        if (calcRight and howmny == HowMny::Back) lacpy(Uplo::General, Q, Vr);
        int info = trevc(side, howmny, select, T, Vl, Vr, work);
        CHECK(info == 0);

        // Now verify the eigenvectors
        for (idx_t j = 0; j < n;) {
            // Get the j-th eigenvalue
            complex_t lambda = w[j];
            bool pair = false;
            if (j < n - 1) {
                if (T(j + 1, j) != TA(0)) {
                    pair = true;
                }
            }

            if (calcRight) {
                // Check the right eigenvector(s)
                std::vector<complex_t> v_;
                auto v = new_vector(v_, n);
                for (idx_t i = 0; i < n; ++i) {
                    v[i] = complex_t(Vr(i, j));
                }
                if (pair) {
                    // Complex conjugate pair
                    for (idx_t i = 0; i < n; ++i) {
                        v[i] = complex_t(Vr(i, j)) +
                               complex_t(0, 1) * complex_t(Vr(i, j + 1));
                    }
                }

                real_t normDiff;
                if (howmny == HowMny::All) {
                    // Compute T * v - lambda * v
                    std::vector<complex_t> Tv_;
                    auto Tv = new_vector(Tv_, n);
                    gemv(Op::NoTrans, one, T, v, zero, Tv);
                    for (idx_t i = 0; i < n; ++i) {
                        Tv[i] -= lambda * v[i];
                    }
                    normDiff = asum(Tv);
                }
                else {
                    // Compute A * v - lambda * v
                    std::vector<complex_t> Av_;
                    auto Av = new_vector(Av_, n);
                    gemv(Op::NoTrans, one, A, v, zero, Av);
                    for (idx_t i = 0; i < n; ++i) {
                        Av[i] -= lambda * v[i];
                    }
                    normDiff = asum(Av);
                }

                real_t normV = asum(v);
                real_t tol_ev = ulp<real_t>() * std::max(real_t(1), normV) *
                                real_t(n) * real_t(10);

                REQUIRE(normDiff <= tol_ev);
            }
            if (calcLeft) {
                // Check the left eigenvector(s)
                std::vector<complex_t> v_;
                auto v = new_vector(v_, n);
                for (idx_t i = 0; i < n; ++i) {
                    v[i] = complex_t(Vl(i, j));
                }
                if (pair) {
                    // Complex conjugate pair
                    if constexpr (is_real<TA>) {
                        for (idx_t i = 0; i < n; ++i) {
                            v[i] = complex_t(Vl(i, j), Vl(i, j + 1));
                        }
                    }
                }

                real_t normDiff;
                if (howmny == HowMny::All) {
                    // Compute v**H * T - lambda * v**H
                    std::vector<complex_t> Tv_;
                    auto Tv = new_vector(Tv_, n);
                    gemv(Op::ConjTrans, one, T, v, zero, Tv);
                    for (idx_t i = 0; i < n; ++i) {
                        Tv[i] -= conj(lambda) * v[i];
                    }
                    normDiff = asum(Tv);
                }
                else {
                    // Compute v**H * A - lambda * v**H
                    std::vector<complex_t> Av_;
                    auto Av = new_vector(Av_, n);
                    gemv(Op::ConjTrans, one, A, v, zero, Av);
                    for (idx_t i = 0; i < n; ++i) {
                        Av[i] -= conj(lambda) * v[i];
                    }
                    normDiff = asum(Av);
                }

                real_t normV = asum(v);
                real_t tol_ev = ulp<real_t>() * std::max(real_t(1), normV) *
                                real_t(n) * real_t(10);

                REQUIRE(normDiff <= tol_ev);
            }

            if (pair) {
                j += 2;
            }
            else {
                j += 1;
            }
        }
    }
}