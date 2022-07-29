/// @file test_unblocked_francis.cpp
/// @brief Test double shift QR algorithm.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("lahqr", "[eigenvalues][doubleshift_qr]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using complex_t = std::complex<real_t>;

    const T zero(0);
    const T one(1);

    auto matrix_type = GENERATE(as<std::string>{}, "Random", "Near overflow");

    idx_t n = 0;
    idx_t ilo = 0;
    idx_t ihi = 0;
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
    std::unique_ptr<T[]> A_(new T[n * n]);
    std::unique_ptr<T[]> H_(new T[n * n]);
    std::unique_ptr<T[]> Q_(new T[n * n]);

    // This only works for legacy matrix, we really work on that construct_matrix function
    auto A = legacyMatrix<T, layout<matrix_t>>(n, n, &A_[0], n);
    auto H = legacyMatrix<T, layout<matrix_t>>(n, n, &H_[0], n);
    auto Q = legacyMatrix<T, layout<matrix_t>>(n, n, &Q_[0], n);

    if (matrix_type == "Random")
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = rand_helper<T>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 2; i < n; ++i)
                A(i, j) = zero;
    }
    if (matrix_type == "Near overflow")
    {
        const real_t large_num = safe_max<real_t>() * uroundoff<real_t>();

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < std::min(n, j + 2); ++i)
                A(i, j) = large_num;

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

    tlapack::lacpy(Uplo::General, A, H);
    auto s = std::vector<complex_t>(n);
    laset(Uplo::General, zero, one, Q);

    DYNAMIC_SECTION("Double shift QR with"
                    << " matrix = " << matrix_type << " n = " << n << " ilo = " << ilo << " ihi = " << ihi)
    {
        int ierr = lahqr(true, true, ilo, ihi, H, s, Q);

        REQUIRE( ierr == 0 );

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
