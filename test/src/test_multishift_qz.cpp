/// @file test_qz_algorithm.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test QZ algorithms.
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
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/gghrd.hpp>
#include <tlapack/lapack/multishift_qz.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QZ algorithm",
                   "[generalized eigenvalues]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;
    using complex_t = complex_type<real_t>;
    using range = pair<idx_t, idx_t>;

    // QZ algorithm does may not work with 16-bit precision types
    if constexpr (sizeof(real_t) <= 2) SKIP_TEST;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    using test_tuple_t = std::tuple<std::string, idx_t>;
    const test_tuple_t test_tuple = GENERATE(
        (test_tuple_t("Large Random", 100)), (test_tuple_t("Random", 0)),
        (test_tuple_t("Random", 1)), (test_tuple_t("Random", 2)),
        (test_tuple_t("Random", 5)), (test_tuple_t("Random", 10)),
        (test_tuple_t("Random", 15)), (test_tuple_t("Random", 20)),
        (test_tuple_t("Random", 30)), (test_tuple_t("Infinite", 10)),
        (test_tuple_t("Infinite", 20)));
    const int seed = GENERATE(2, 3);

    const std::string matrix_type = std::get<0>(test_tuple);
    const idx_t n = std::get<1>(test_tuple);
    const idx_t ilo = 0;
    const idx_t ihi = n;
    const real_t zero(0);
    const real_t one(1);

    // Seed random number generator
    mm.gen.seed(seed);

    // Define the matrices
    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<TA> B_;
    auto B = new_matrix(B_, n, n);
    std::vector<TA> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);
    std::vector<TA> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<TA> Z_;
    auto Z = new_matrix(Z_, n, n);

    if (matrix_type == "Random") {
        mm.hessenberg(A);
        mm.random(Uplo::Upper, B);
    }
    if (matrix_type == "Large Random") {
        // Generate full matrix
        mm.random(A);
        mm.random(B);
        // Hessenberg triangular factorization
        std::vector<TA> tau(n);
        geqrf(B, tau);
        unmqr(LEFT_SIDE, CONJ_TRANS, B, tau, A);
        gghrd(false, false, ilo, ihi, A, B, Q, Z);
    }
    if (matrix_type == "Infinite") {
        // Generate pencil with infinite eigenvalues
        mm.hessenberg(A);
        mm.random(Uplo::Upper, B);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < n; ++i)
                B(i, j) = zero;
        if (n > 4) B(4, 4) = zero;
        if (n > 7) B(7, 7) = zero;
        if (n > 14) B(14, 14) = zero;
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (TA)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (TA)0.0;

    // Clear out subdiagonal
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = zero;
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            B(i, j) = zero;

    lacpy(GENERAL, A, H);
    lacpy(GENERAL, B, T);
    std::vector<complex_t> alpha(n);
    std::vector<TA> beta(n);
    laset(GENERAL, zero, one, Q);
    laset(GENERAL, zero, one, Z);

    DYNAMIC_SECTION("matrix = " << matrix_type << " n = " << n
                                << " ilo = " << ilo << " ihi = " << ihi
                                << " seed = " << seed)
    {
        FrancisOpts opts;
        opts.nmin = 15;

        int ierr = multishift_qz(true, true, true, ilo, ihi, H, T, alpha, beta,
                                 Q, Z, opts);
        CHECK(ierr == 0);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                H(i, j) = zero;
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < n; ++i)
                T(i, j) = zero;

        const real_t eps = uroundoff<real_t>();
        const real_t tol = real_t(n * 1.0e2) * eps;

        std::vector<TA> res_;
        auto res = new_matrix(res_, n, n);
        std::vector<TA> work_;
        auto work = new_matrix(work_, n, n);

        // Calculate residuals
        auto orth_res_norm_q = check_orthogonality(Q, res);
        CHECK(orth_res_norm_q <= tol);

        auto orth_res_norm_z = check_orthogonality(Z, res);
        CHECK(orth_res_norm_z <= tol);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normA_res =
            check_generalized_similarity_transform(A, Q, Z, H, res, work);
        CHECK(normA_res <= tol * normA);

        auto normB = tlapack::lange(tlapack::FROB_NORM, B);
        auto normB_res =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);
        CHECK(normB_res <= tol * normB);

        // Check that the eigenvalues match with the diagonal elements
        // @todo : also check normalization
        idx_t i = ilo;
        while (i < ihi) {
            int nb = 1;
            if (is_real<TA>)
                if (i + 1 < ihi)
                    if (H(i + 1, i) != zero) nb = 2;

            if (nb == 1) {
                CHECK(abs1(alpha[i] - H(i, i)) <=
                      tol * max(real_t(1), abs1(H(i, i))));
                CHECK(abs1(beta[i] - T(i, i)) <=
                      tol * max(real_t(1), abs1(T(i, i))));
                i = i + 1;
            }
            else {
                TA beta1, beta2;
                complex_t alpha1, alpha2;
                auto H22 = slice(H, range(i, i + 2), range(i, i + 2));
                auto T22 = slice(T, range(i, i + 2), range(i, i + 2));
                lahqz_eig22(H22, T22, alpha1, alpha2, beta1, beta2);
                if (abs1(alpha1 - alpha[i]) > abs1(alpha2 - alpha[i])) {
                    auto swp1 = alpha1;
                    alpha1 = alpha2;
                    alpha2 = swp1;

                    auto swp2 = beta1;
                    beta1 = beta2;
                    beta2 = swp2;
                }
                CHECK(abs1(alpha[i] - alpha1) <=
                      tol * max(real_t(1), abs1(alpha1)));
                CHECK(abs1(beta[i] - beta1) <=
                      tol * max(real_t(1), abs1(beta1)));
                CHECK(abs1(alpha[i + 1] - alpha2) <=
                      tol * max(real_t(1), abs1(alpha2)));
                CHECK(abs1(beta[i + 1] - beta2) <=
                      tol * max(real_t(1), abs1(beta2)));
                i = i + 2;
            }
        }
    }
}
