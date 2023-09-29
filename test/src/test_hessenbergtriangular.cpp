/// @file test_hessenbergtriangular.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test hessenberg triangular reduction
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
#include <tlapack/lapack/ungqr.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "Hessenberg-triangular reduction is backward stable",
    "[eigenvalues][generalized eigenvalues][hessenbergtriangular]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<TA>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const std::string matrix_type = GENERATE("Near_overflow", "Random");
    const idx_t n = GENERATE(1, 2, 3, 5, 10);
    const idx_t ilo_offset = GENERATE(0, 1);
    const idx_t ihi_offset = GENERATE(0, 1);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(10. * n) * eps;

    // Only runs the near overflow case
    // when n = 5 and ilo_offset = 0 and ihi_offset = 0
    if (matrix_type == "Near_overflow" && n != 5 && ilo_offset != 0 &&
        ihi_offset != 0)
        SKIP_TEST;

    // Constants
    const idx_t ilo = n > 1 ? ilo_offset : 0;
    const idx_t ihi = n > 1 + ilo_offset ? n - ihi_offset : n;

    // Define the matrices and vectors
    std::vector<TA> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<TA> B_;
    auto B = new_matrix(B_, n, n);
    std::vector<TA> Q_;
    auto Q = new_matrix(Q_, n, n);
    std::vector<TA> Z_;
    auto Z = new_matrix(Z_, n, n);
    std::vector<TA> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<TA> T_;
    auto T = new_matrix(T_, n, n);
    std::vector<TA> tau(n);

    if (matrix_type == "Random") {
        mm.random(A);
        mm.random(B);
    }
    else if (matrix_type == "Near_overflow") {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();
        mm.single_value(A, large_num);
        mm.random(B);
    }
    else if (matrix_type == "stdin") {
        mm.colmajor_read(A, std::cin);
        mm.colmajor_read(B, std::cin);
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            A(i, j) = (TA)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i; ++j)
            A(i, j) = (TA)0.0;
    for (idx_t j = 0; j < ilo; ++j)
        for (idx_t i = j; i < n; ++i)
            B(i, j) = (TA)0.0;
    for (idx_t i = ihi; i < n; ++i)
        for (idx_t j = 0; j < i + 1; ++j)
            B(i, j) = (TA)0.0;

    // Copy matrix A and B to H and T so we can check the error later
    tlapack::lacpy(GENERAL, A, H);
    tlapack::lacpy(GENERAL, B, T);

    // QR factorization of B
    geqrf(T, tau);
    unmqr(LEFT_SIDE, CONJ_TRANS, T, tau, H);
    tlapack::lacpy(GENERAL, T, Q);
    ungqr(Q, tau);
    laset(GENERAL, (TA)0, (TA)1, Z);

    DYNAMIC_SECTION("matrix = " << matrix_type << " n = " << n
                                << " ilo = " << ilo << " ihi = " << ihi)
    {
        gghrd(true, true, ilo, ihi, H, T, Q, Z);

        // Check orthogonality
        auto orth_Q = check_orthogonality(Q);
        CHECK(orth_Q <= tol);
        auto orth_Z = check_orthogonality(Z);
        CHECK(orth_Z <= tol);

        // Check backward stability
        std::vector<TA> res_;
        auto res = new_matrix(res_, n, n);
        std::vector<TA> work_;
        auto work = new_matrix(work_, n, n);

        auto normA = tlapack::lange(tlapack::FROB_NORM, A);
        auto normA_res =
            check_generalized_similarity_transform(A, Q, Z, H, res, work);
        CHECK(normA_res <= tol * normA);

        auto normB = tlapack::lange(tlapack::FROB_NORM, B);
        auto normB_res =
            check_generalized_similarity_transform(B, Q, Z, T, res, work);
        CHECK(normB_res <= tol * normB);
    }
}
