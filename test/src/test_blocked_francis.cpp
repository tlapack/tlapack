/// @file test_blocked_francis.cpp
/// @brief Test multishift QR algorithm.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <tlapack/plugins/stdvector.hpp>
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

    // Functor
    Create<matrix_t> new_matrix;

    rand_generator gen;

    const T zero(0);
    const T one(1);

    auto matrix_type = GENERATE(as<std::string>{}, "Large Random", "Random");
    // The near overflow tests are disabled untill a bug in rotg is fixed
    // auto matrix_type = GENERATE(as<std::string>{}, "Large Random", "Near overflow", "Random");

    idx_t n = 0;
    idx_t ilo = 0;
    idx_t ihi = 0;
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
    if (matrix_type == "Large Random")
    {
        n = 100;
        ilo = 0;
        ihi = n;
    }

    // Define the matrices and vectors
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> H_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[n * n]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = new_matrix( &A_[0], n, n );
    auto H = new_matrix( &H_[0], n, n );
    auto Q = new_matrix( &Q_[0], n, n );

    if (matrix_type == "Random")
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = rand_helper<T>(gen);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * ulp<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = large_num;

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Large Random")
    {
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

    lacpy(Uplo::General, A, H);
    auto s = std::vector<complex_t>(n);
    laset(Uplo::General, zero, one, Q);

    idx_t ns = GENERATE(4, 2);
    idx_t nw = GENERATE(4, 2);

    DYNAMIC_SECTION("Multishift QR with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi << " ns = " << ns << " nw = " << nw << " seed = " << seed)
    {

        francis_opts_t<> opts;
        opts.nshift_recommender = [ns](idx_t n, idx_t nh) -> idx_t
        {
            return ns;
        };
        opts.deflation_window_recommender = [nw](idx_t n, idx_t nh) -> idx_t
        {
            return nw;
        };
        opts.nmin = 15;

        int ierr = multishift_qr(true, true, ilo, ihi, H, s, Q, opts);

        CHECK(ierr == 0);

        // Clean the lower triangular part that was used a workspace
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                H(i, j) = zero;

        const real_type<T> eps = uroundoff<real_type<T>>();
        const real_type<T> tol = n * 1.0e2 * eps;

        std::unique_ptr<T[]> _res(new T[n * n]);
        std::unique_ptr<T[]> _work(new T[n * n]);

        auto res = new_matrix( &_res[0], n, n );
        auto work = new_matrix( &_work[0], n, n );

        // Calculate residuals
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);

        auto normA = tlapack::lange(tlapack::frob_norm, A);
        auto simil_res_norm = check_similarity_transform(A, Q, H, res, work);
        CHECK(simil_res_norm <= tol * normA);

        // Check that the eigenvalues match with the diagonal elements
        idx_t i = ilo;
        while (i < ihi)
        {
            int nb = 1;
            if (!is_complex<T>::value)
                if (i + 1 < ihi)
                    if (H(i + 1, i) != zero)
                        nb = 2;

            if (nb == 1)
            {
                CHECK( abs1( s[i] - H(i,i) ) <= tol * std::max<real_t>(1,abs1(H(i,i))) );
                i = i + 1;
            } else {

                T a11, a12, a21, a22, sn;
                real_t cs;
                a11 = H(i,i);
                a12 = H(i,i+1);
                a21 = H(i+1,i);
                a22 = H(i+1,i+1);
                complex_t s1, s2, swp;
                lahqr_schur22( a11, a12, a21, a22, s1, s2, cs, sn );
                if( abs1( s1 - s[i] ) > abs1( s2 - s[i] ) ){
                    swp = s1;
                    s1 = s2;
                    s2 = swp;
                }
                CHECK( abs1( s[i] - s1 ) <= tol * std::max<real_t>(1,abs1(s1)) );
                CHECK( abs1( s[i+1] - s2 ) <= tol * std::max<real_t>(1,abs1(s2)) );
                i = i + 2;
            }
        }
    }
}
