/// @file test_qr_algorithm.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @brief Test QR algorithms.
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
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/qr_iteration.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("QR algorithm",
                   "[eigenvalues][doubleshift_qr][multishift_qr]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    using test_tuple_t = std::tuple<std::string, idx_t>;
    const test_tuple_t test_tuple = GENERATE(
        (test_tuple_t("Near overflow", 4)), (test_tuple_t("Near overflow", 10)),
        (test_tuple_t("Large Random", 100)), (test_tuple_t("Random", 0)),
        (test_tuple_t("Random", 1)), (test_tuple_t("Random", 2)),
        (test_tuple_t("Random", 5)), (test_tuple_t("Random", 10)),
        (test_tuple_t("Random", 15)), (test_tuple_t("Random", 20)),
        (test_tuple_t("Random", 30)));
    const int seed = GENERATE(2, 3);

    using variant_t = std::tuple<QRIterationVariant, idx_t, idx_t>;
    const variant_t variant =
        GENERATE((variant_t(QRIterationVariant::DoubleShift, 0, 0)),
                 (variant_t(QRIterationVariant::MultiShift, 4, 4)),
                 (variant_t(QRIterationVariant::MultiShift, 4, 2)),
                 (variant_t(QRIterationVariant::MultiShift, 2, 4)),
                 (variant_t(QRIterationVariant::MultiShift, 2, 2)));

    const std::string matrix_type = std::get<0>(test_tuple);
    const idx_t n = std::get<1>(test_tuple);
    const idx_t ilo = 0;
    const idx_t ihi = n;
    const real_t zero(0);
    const real_t one(1);
    const idx_t ns = std::get<1>(variant);
    const idx_t nw = std::get<2>(variant);

    // Only run the large random test once
    if (matrix_type == "Large Random" && seed != 2) SKIP_TEST;

    // Only run the large random if we are testing multishift qr
    if (matrix_type == "Large Random" &&
        std::get<0>(variant) != QRIterationVariant::MultiShift)
        SKIP_TEST;

    // Random number generator
    PCG32 gen;
    gen.seed(seed);

    // Define the matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);

    if (matrix_type == "Random") {
        mm.hessenberg(A);
    }
    if (matrix_type == "Near overflow") {
        const real_t large_num = safe_max<real_t>() * ulp<real_t>();
        mm.single_value(A, large_num);
    }
    if (matrix_type == "Large Random") {
        // Generate full matrix
        mm.random(A);

        // Hessenberg factorization
        std::vector<T> tau(n);
        gehrd(0, n, A, tau);
    }

    // Throw away reflectors
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            A(i, j) = zero;

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;

    lacpy(GENERAL, A, H);
    std::vector<complex_t> s(n);
    laset(GENERAL, zero, one, Q);

    DYNAMIC_SECTION("matrix = " << matrix_type << " n = " << n << " ilo = "
                                << ilo << " ihi = " << ihi << " ns = " << ns
                                << " nw = " << nw << " seed = " << seed
                                << " variant = " << (char)std::get<0>(variant))
    {
        QRIterationOpts opts;
        opts.variant = std::get<0>(variant);
        opts.nshift_recommender = [ns](idx_t n, idx_t nh) -> idx_t {
            return ns;
        };
        opts.deflation_window_recommender = [nw](idx_t n, idx_t nh) -> idx_t {
            return nw;
        };
        opts.nmin = 15;

        int ierr = qr_iteration(true, true, ilo, ihi, H, s, Q, opts);
        CHECK(ierr == 0);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                H(i, j) = zero;

        const real_t eps = uroundoff<real_t>();
        const real_t tol = real_t(n * 1.0e2) * eps;

        std::vector<T> res_;
        auto res = new_matrix(res_, n, n);
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol * normA);

        // Check that the eigenvalues match with the diagonal elements
        idx_t i = ilo;
        while (i < ihi) {
            int nb = 1;
            if constexpr (is_real<T>)
                if (i + 1 < ihi)
                    if (H(i + 1, i) != zero) nb = 2;

            if (nb == 1) {
                CHECK(abs1(s[i] - H(i, i)) <=
                      tol * max(real_t(1), abs1(H(i, i))));
                i = i + 1;
            }
            else {
                T a11, a12, a21, a22, sn;
                real_t cs;
                a11 = H(i, i);
                a12 = H(i, i + 1);
                a21 = H(i + 1, i);
                a22 = H(i + 1, i + 1);
                complex_t s1, s2, swp;
                if constexpr (is_real<T>)
                    lahqr_schur22(a11, a12, a21, a22, s1, s2, cs, sn);
                if (abs1(s1 - s[i]) > abs1(s2 - s[i])) {
                    swp = s1;
                    s1 = s2;
                    s2 = swp;
                }
                CHECK(abs1(s[i] - s1) <= tol * max(real_t(1), abs1(s1)));
                CHECK(abs1(s[i + 1] - s2) <= tol * max(real_t(1), abs1(s2)));
                i = i + 2;
            }
        }
    }
}
