/// @file test_blocked_francis.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test multishift QR algorithm.
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
#include <tlapack/lapack/gehrd.hpp>
#include <tlapack/lapack/multishift_qr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Multishift QR",
                   "[eigenvalues][multishift_qr]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = complex_type<real_t>;

    // Functor
    Create<matrix_t> new_matrix;

    using test_tuple_t = std::tuple<std::string, idx_t>;
    const test_tuple_t test_tuple = GENERATE(
        (test_tuple_t("Large Random", 100)), (test_tuple_t("Random", 15)),
        (test_tuple_t("Random", 20)), (test_tuple_t("Random", 30)));
    // The near overflow tests are disabled untill a bug in rotg is fixed
    // const std::string matrix_type = GENERATE(as<std::string>{}, "Large
    // Random", "Near overflow", "Random");

    const std::string matrix_type = std::get<0>(test_tuple);
    const idx_t n = std::get<1>(test_tuple);

    const int seed = GENERATE(2, 3, 4, 5, 6, 7, 8, 9, 10);

    // Only run the large random test once
    if (matrix_type == "Large Random" && seed != 2) return;

    // Constants
    const idx_t ilo = 0;
    const idx_t ihi = n;
    const real_t zero(0);
    const real_t one(1);

    // Random number generator
    rand_generator gen;
    gen.seed(seed);

    // Define the matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);

    if (matrix_type == "Random") {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < min(n, j + 2); ++i)
                A(i, j) = rand_helper<T>(gen);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Near overflow") {
        const real_t large_num = safe_max<real_t>() * ulp<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < min(n, j + 2); ++i)
                A(i, j) = large_num;

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Large Random") {
        // Generate full matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                A(i, j) = rand_helper<T>(gen);

        // Hessenberg factorization
        std::vector<T> tau(n);
        gehrd(0, n, A, tau);

        // Throw away reflectors
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }

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

    idx_t ns = GENERATE(4, 2);
    idx_t nw = GENERATE(4, 2);

    DYNAMIC_SECTION("matrix = " << matrix_type << " n = " << n << " ilo = "
                                << ilo << " ihi = " << ihi << " ns = " << ns
                                << " nw = " << nw << " seed = " << seed)
    {
        FrancisOpts opts;
        opts.nshift_recommender = [ns](idx_t n, idx_t nh) -> idx_t {
            return ns;
        };
        opts.deflation_window_recommender = [nw](idx_t n, idx_t nh) -> idx_t {
            return nw;
        };
        opts.nmin = 15;

        int ierr = multishift_qr(true, true, ilo, ihi, H, s, Q, opts);

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
            if (is_real<T>)
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
