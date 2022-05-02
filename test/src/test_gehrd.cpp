/// @file test_gehrd.cpp
/// @brief Test hessenberg reduction
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_debugutils.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <iostream>
#include <iomanip>

using namespace tlapack;

TEMPLATE_TEST_CASE("Hessenberg reduction is backward stable", "[eigenvalues]", float, double, std::complex<float>, std::complex<double>)
{
    srand(1);

    using T = TestType;
    using idx_t = std::size_t;
    using real_t = real_type<T>;
    using tlapack::internal::colmajor_matrix;
    using pair = pair<idx_t, idx_t>;

    auto matrix_type = GENERATE("Random", "Near overflow");

    idx_t n, ilo, ihi;
    if (matrix_type == "Random")
    {
        // Generate n
        n = GENERATE(1, 2, 3, 5, 10);
        // Generate ilo and ihi
        idx_t ilo_offset = GENERATE(0, 1);
        idx_t ihi_offset = GENERATE(0, 1);
        ilo = n > 1 ? ilo_offset : 0;
        ihi = n > 1 + ilo_offset ? n - ihi_offset : n;
    }
    if (matrix_type == "Near overflow")
    {
        n = 5;
        ilo = 0;
        ihi = n;
    }

    const real_t eps = uroundoff<real_t>();
    const real_t tol = n * 1.0e2 * eps;

    // Define the matrices and vectors
    std::unique_ptr<T[]> _A(new T[n * n]);
    std::unique_ptr<T[]> _H(new T[n * n]);
    std::unique_ptr<T[]> _Q(new T[n * n]);
    std::unique_ptr<T[]> _res(new T[n * n]);
    std::unique_ptr<T[]> _work(new T[n * n]);

    auto A = colmajor_matrix<T>(&_A[0], n, n);
    auto H = colmajor_matrix<T>(&_H[0], n, n);
    auto Q = colmajor_matrix<T>(&_Q[0], n, n);
    auto res = colmajor_matrix<T>(&_res[0], n, n);
    auto work = colmajor_matrix<T>(&_work[0], n, n);
    std::vector<T> tau(n);
    std::vector<T> workv(n);

    if (matrix_type == "Random")
    {
        // Generate a random matrix in A
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                A(i, j) = large_num;
    }

    // Make sure ilo and ihi correspond to the actual matrix
    for (size_t j = 0; j < ilo; ++j)
        for (size_t i = j + 1; i < n; ++i)
            A(i, j) = (T)0.0;
    for (size_t i = ihi; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            A(i, j) = (T)0.0;
    tlapack::lacpy(Uplo::General, A, H);

    DYNAMIC_SECTION("GEHD2 with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi)
    {
        tlapack::gehd2(ilo, ihi, H, tau, workv);

        // Generate orthogonal matrix Q
        tlapack::lacpy(Uplo::General, H, Q);
        tlapack::unghr(ilo, ihi, Q, tau, workv);

        // Remove junk from lower half of H
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                H(i, j) = 0.0;

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol);
    }
    idx_t nb = GENERATE(2, 3);
    DYNAMIC_SECTION("GEHRD with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi << " nb = " << nb)
    {
        gehrd_opts_t<idx_t, T> opts = {.nb = nb, .nx_switch = 2};
        idx_t required_workspace = get_work_gehrd(ilo, ihi, A, tau, opts);
        std::unique_ptr<T[]> _work2(new T[required_workspace]);
        opts._work = &_work2[0];
        opts.lwork = required_workspace;
        tlapack::gehrd(ilo, ihi, H, tau, opts);

        // Generate orthogonal matrix Q
        tlapack::lacpy(Uplo::General, H, Q);
        tlapack::unghr(ilo, ihi, Q, tau, workv);

        // Remove junk from lower half of H
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j + 2; i < n; ++i)
                H(i, j) = 0.0;

        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol);
    }
}