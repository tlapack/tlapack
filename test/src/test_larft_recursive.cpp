/// @file test_larft_recursive.cpp
/// @brief Test for larft recursive
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <plugins/tlapack_legacyArray.hpp>
#include <lapack/larft_recursive.hpp>
#include <testutils.hpp>
#include <testdefinitions.hpp>

using namespace tlapack;

/// TODO: Improve the tad here: 
TEMPLATE_LIST_TEST_CASE("larft_recursive works properly", "[qr]", types_to_test)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    idx_t m = GENERATE(10, 5);
    // Once 1x2 solver is finished, generate n independantly
    idx_t n = GENERATE( 6, 8 );

    const idx_t k = std::min(m,n);

    const real_t eps = uroundoff<real_t>();
    const real_t tol = 1.0e2 * eps;

    std::unique_ptr<T[]> A_(new T[m * n]);
    std::unique_ptr<T[]> B_(new T[m * n]);
    std::unique_ptr<T[]> TT_(new T[k * k]);
    std::unique_ptr<T[]> TTT_(new T[k * k]);

    std::vector<T> tau(k);
    std::vector<T> work(std::max(m,n));

    auto A = legacyMatrix<T, layout<matrix_t>>(m, n, &A_[0], m);
    auto B = legacyMatrix<T, layout<matrix_t>>(m, n, &B_[0], m);
    auto TT = legacyMatrix<T, layout<matrix_t>>(k, k, &TT_[0], k);
    auto TTT = legacyMatrix<T, layout<matrix_t>>(k, k, &TTT_[0], k);

    for (idx_t i = 0; i < m; ++i)
        for (idx_t j = 0; j < n; ++j)
            A(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    
    for (idx_t i = 0; i < m; ++i)
        for (idx_t j = 0; j < n; ++j)
            B(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (idx_t i = 0; i < k; ++i)
        for (idx_t j = 0; j < k; ++j)
            TT(i, j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
 
     for (idx_t i = 0; i < k; ++i) {
        for (idx_t j = 0; j < k; ++j)
            TTT(i,j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        for (idx_t j = 0; j < i; ++j)
            TTT(i,j) = T(-1);
    }

    
    // Op trans_l = Op::NoTrans;
    // Op trans_r = Op::NoTrans;

    // int sign = 1;

    // // Calculate op(TL)*X + ISGN*X*op(TR)
    // gemm(trans_l, Op::NoTrans, one, TL, X_exact, zero, B);
    // gemm(Op::NoTrans, trans_r, sign, X_exact, TR, one, B);

    tlapack::geqr2( A, tau, work );

    tlapack::larft(tlapack::Direction::Forward, tlapack::columnwise_storage, A, tau, TT);




    DYNAMIC_SECTION("m = " << m << " n =" << n)
    {

        larft_recursive( A, tau, TTT);

        // TTT receives TTT - TT on the upper part
        // The strict lower part of TTT receives 0's
        for (idx_t j = 0; j < k; ++j){ 
            for (idx_t i = 0; i <= j; ++i)
                TTT(i,j) -= TT(i,j);
                
            // for (idx_t i = j+1; i <k; ++i)
            //     TTT(i,j) = tlapack::make_scalar<T>(0,0);
        }
        // // Strict lower part of TT will be 0's 
        // for (idx_t j = 0; j < k; ++j){    
        //     for (idx_t i = j+1; i <k; ++i)
        //         TT(i,j) = tlapack::make_scalar<T>(0,0);
        // }

        // lantr
        real_t norm = lantr( tlapack::max_norm, tlapack::Uplo::Upper, tlapack::Diag::NonUnit, TTT )
         / lantr( tlapack::max_norm, tlapack::Uplo::Upper, tlapack::Diag::NonUnit,
          TT );


        //check if the lower part is -1
        bool lowerNotFound = true;
        for (idx_t i = 0; i < k; ++i) {
            for (idx_t j = 0; j < i; ++j){
                if(TTT(i,j) != T(-1))
                    lowerNotFound = false;
            }    
        }


        // Relative Error
        //real_t norm = tlapack::lange(tlapack::max_norm, TTT) / tlapack::lange(tlapack::max_norm, TT);

        CHECK( norm <= tol );
        CHECK( lowerNotFound == true );


    }
}
