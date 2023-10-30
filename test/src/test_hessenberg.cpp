/// @file test_hessenberg.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test hessenberg reduction
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

// Other routines
#include <tlapack/lapack/hessenberg.hpp>
#include <tlapack/lapack/unghr.hpp>

using namespace tlapack;

template <typename matrix_t, typename vector_t>
void check_hess_reduction(size_type<matrix_t> ilo,
                          size_type<matrix_t> ihi,
                          matrix_t H,
                          vector_t tau,
                          matrix_t A)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    idx_t n = ncols(A);

    const real_type<T> eps = uroundoff<real_type<T>>();
    const real_type<T> tol = real_type<T>(n * 1.0e2) * eps;

    std::vector<T> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<T> res_;
    auto res = new_matrix(res_, n, n);
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);

    // Generate orthogonal matrix Q
    tlapack::lacpy(GENERAL, H, Q);
    tlapack::unghr(ilo, ihi, Q, tau);

    // Remove junk from lower half of H
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 2; i < n; ++i)
            H(i, j) = T(0);

    // Calculate residuals
    auto orth_res_norm = check_orthogonality(Q, res);
    CHECK(orth_res_norm <= tol);

    auto normA = tlapack::lange(tlapack::FROB_NORM, A);
    auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
    CHECK(simil_res_norm <= tol * normA);
}

TEMPLATE_TEST_CASE("Hessenberg reduction is backward stable",
                   "[eigenvalues][hessenberg]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    using variant_t = pair<HessenbergVariant, idx_t>;
    const variant_t variant =
        GENERATE((variant_t(HessenbergVariant::Blocked, 2)),
                 (variant_t(HessenbergVariant::Blocked, 3)),
                 (variant_t(HessenbergVariant::Level2, 1)));
    const std::string matrix_type = GENERATE("Near_overflow", "Random");
    const idx_t n = GENERATE(1, 2, 3, 5, 10);
    const idx_t ilo_offset = GENERATE(0, 1);
    const idx_t ihi_offset = GENERATE(0, 1);

    // Only runs the near overflow case
    // when n = 5 and ilo_offset = 0 and ihi_offset = 0
    if (matrix_type == "Near_overflow" && n != 5 && ilo_offset != 0 &&
        ihi_offset != 0)
        SKIP_TEST;

    // Constants
    const idx_t ilo = n > 1 ? ilo_offset : 0;
    const idx_t ihi = n > 1 + ilo_offset ? n - ihi_offset : n;

    // Define the matrices and vectors
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> tau(n);

    if (matrix_type == "Random") {
        mm.random(A);
    }
    else if (matrix_type == "Near_overflow") {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();
        mm.single_value(A, large_num);
    }
    else if (matrix_type == "stdin") {
        mm.colmajor_read(A, std::cin);
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;
    tlapack::lacpy(GENERAL, A, H);

    DYNAMIC_SECTION("matrix = " << matrix_type << " n = " << n
                                << " ilo = " << ilo << " ihi = " << ihi
                                << " variant = " << (char)variant.first
                                << " nb = " << variant.second)
    {
        HessenbergOpts opts;
        opts.nb = variant.second;
        opts.nx_switch = 2;
        hessenberg(ilo, ihi, H, tau, opts);

        check_hess_reduction(ilo, ihi, H, tau, A);
    }
}
