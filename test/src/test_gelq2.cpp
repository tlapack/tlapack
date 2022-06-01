/// @file test_gelq2.cpp
/// @brief Test 1x1 and 2x2 sylvester solver
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <tlapack.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

TEMPLATE_LIST_TEST_CASE("LQ factorization of a general m-by-n matrix", "[lqf]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    const T zero(0);
    const T one(1);

    idx_t m, n, k;

    bool bidg = false;

    // m = GENERATE( 10, 20, 30 );
    // n = GENERATE( 10, 20, 30 );
    // k = GENERATE( 8, 20, 30);
    m = 10;
    n = 10;
    k = 8;

    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * max(m,n) * eps;

    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> A_copy_(new T[m * n]);

    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], m );
    // auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], layout<matrix_t> == Layout::ColMajor ? m : n );
    // auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], m  );
    // auto A_copy = legacyMatrix<T, layout<matrix_t>>(m, n, &A_copy_[0], layout<matrix_t> == Layout::ColMajor ? m : n  );

    std::vector<T> work(max(m,n)); // max of m and n
    std::vector<T> tauw(min(m,n)); // min of m and n

    // Generate random m-by-n matrix
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // lacpy(Uplo::General, A, A_copy);

    DYNAMIC_SECTION("m = " << m << " n = " << n << " k = " << k)
    {
        gelq2(A, tauw, work);

        std::unique_ptr<T[]> L_(new T[k * n]);
        auto L = legacyMatrix<T, layout<matrix_t>>(k, n, &L_[0], k );

        lacpy(Uplo::General, slice(A, range(0, std::min(m, k)), range (0, n)), L); 

        ungl2(L, tauw, work, bidg);

        auto Q = slice( L, range(0,k), range(0,n) );

        std::unique_ptr<T[]> _res(new T[k * k]);

        auto res = legacyMatrix<T, layout<matrix_t>>(k, k, &_res[0], k);
        auto orth_res_norm = check_orthogonality(Q, res);
        CHECK(orth_res_norm <= tol);


    }

}