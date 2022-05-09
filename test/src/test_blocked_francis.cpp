/// @file test_blocked_francis.cpp
/// @brief Test utils from <T>LAPACK.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("Multishift QR", "[eigenvalues][multishift_qr]", types_to_test)
{

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = std::complex<real_t>;

    rand_generator gen;

    const T zero(0);
    const T one(1);

    auto matrix_type = GENERATE(as<std::string>{}, "Near overflow", "Random");

    idx_t n, ilo, ihi;
    int seed = 0;
    if (matrix_type == "Random")
    {
        seed = GENERATE(2, 3, 4, 5, 6, 7, 8, 9, 10);
        gen.seed(seed);
        // Generate n
        n = GENERATE(15, 20, 30);
        ilo = 0;
        ihi = n;
    }
    if (matrix_type == "Near overflow")
    {
        n = 30;
        ilo = 0;
        ihi = n;
    }

    // Define the matrices and vectors
    std::unique_ptr<T[]> _A(new T[n * n]);
    std::unique_ptr<T[]> _H(new T[n * n]);
    std::unique_ptr<T[]> _Q(new T[n * n]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &_A[0], n);
    auto H = legacyMatrix<T, layout<matrix_t>>(n, n, &_H[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n, n, &_Q[0], n);

    if (matrix_type == "Random")
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = rand_helper<T>(gen);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = large_num;

        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (size_t j = 0; j < ilo; ++j)
        for (size_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (size_t i = ihi; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;

    lacpy(Uplo::General, A, H);
    auto s = std::vector<complex_t>(n);
    laset(Uplo::General, zero, one, Q);

    idx_t ns = GENERATE(4, 2);
    idx_t nw = GENERATE(4, 2);

    DYNAMIC_SECTION("Multishift QR with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi << " ns = " << ns << " nw = " << nw << " seed = " << seed)
    {

        francis_opts_t<idx_t, T> opts;
        opts.nshift_recommender = [ns](idx_t n, idx_t nh) -> idx_t
        {
            return ns;
        };
        opts.deflation_window_recommender = [nw](idx_t n, idx_t nh) -> idx_t
        {
            return nw;
        };

        int ierr = multishift_qr(true, true, ilo, ihi, H, s, Q, opts);

        CHECK(ierr == 0);

        // Clean the lower triangular part that was used a workspace
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                H(i, j) = zero;

        const real_type<T> eps = uroundoff<real_type<T>>();
        const real_type<T> tol = n * 1.0e2 * eps;

        std::unique_ptr<T[]> _res(new T[n * n]);
        std::unique_ptr<T[]> _work(new T[n * n]);

        auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &_res[0], n);
        auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &_work[0], n);

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::frob_norm, A);
        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol * normA);
    }
}

TEMPLATE_LIST_TEST_CASE("lahqr", "[eigenvalues]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = std::complex<real_t>;

    const T zero(0);
    const T one(1);

    auto matrix_type = GENERATE(as<std::string>{}, "Near overflow", "Random");

    idx_t n, ilo, ihi;
    if (matrix_type == "Random")
    {
        // Generate n
        n = GENERATE(0, 1, 2, 5, 10, 15);
        ilo = 0;
        ihi = n;
    }
    if (matrix_type == "Near overflow")
    {
        n = GENERATE(4, 10);
        ilo = 0;
        ihi = n;
    }

    // Define the matrices and vectors
    std::unique_ptr<T[]> _A(new T[n * n]);
    std::unique_ptr<T[]> _H(new T[n * n]);
    std::unique_ptr<T[]> _Q(new T[n * n]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &_A[0], n);
    auto H = legacyMatrix<T, layout<matrix_t>>(n, n, &_H[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n, n, &_Q[0], n);

    if (matrix_type == "Random")
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = rand_helper<T>();

        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = large_num;

        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (size_t j = 0; j < ilo; ++j)
        for (size_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (size_t i = ihi; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;

    tlapack::lacpy(Uplo::General, A, H);
    auto s = std::vector<complex_t>(n);
    laset(Uplo::General, zero, one, Q);

    DYNAMIC_SECTION("Double shift QR with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi)
    {
        lahqr(true, true, ilo, ihi, H, s, Q);

        const real_type<T> eps = uroundoff<real_type<T>>();
        const real_type<T> tol = n * 1.0e2 * eps;

        std::unique_ptr<T[]> _res(new T[n * n]);
        std::unique_ptr<T[]> _work(new T[n * n]);

        auto res = legacyMatrix<T, layout<matrix_t>>(n, n, &_res[0], n);
        auto work = legacyMatrix<T, layout<matrix_t>>(n, n, &_work[0], n);

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::frob_norm, A);
        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol * normA);
    }
}