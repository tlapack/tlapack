/// @file test_lasy2.cpp
/// @brief Test 1x1 and 2x2 sylvester solver
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of testBLAS.
// testBLAS is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <catch2/catch.hpp>
#include <tlapack.hpp>
#include <iostream>
#include <iomanip>

using namespace tlapack;

// This should really be moved to test utils or something
template <typename matrix_t>
inline void printMatrix(const matrix_t &A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i)
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(16) << A(i, j) << " ";
    }
}

TEST_CASE( "1x1 sylvester solver gives correct result", "[utils]" ) {

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n1 = 1;
    const idx_t n2 = 1;
    const real_t eps = uroundoff<real_t>();

    Op trans_l = Op::NoTrans;
    Op trans_r = Op::NoTrans;

    std::unique_ptr<T[]> _TL(new T[n1*n1]);
    auto TL = colmajor_matrix<T>(&_TL[0], n1, n1);

    std::unique_ptr<T[]> _TR(new T[n2*n2]);
    auto TR = colmajor_matrix<T>(&_TR[0], n2, n2);

    std::unique_ptr<T[]> _B(new T[n1*n2]);
    auto B = colmajor_matrix<T>(&_B[0], n1, n2);

    std::unique_ptr<T[]> _X(new T[n1 * n2]);
    auto X = colmajor_matrix<T>(&_X[0], n1, n2);

    std::unique_ptr<T[]> _X_exact(new T[n1 * n2]);
    auto X_exact = colmajor_matrix<T>(&_X_exact[0], n1, n2);

    TL(0,0) = 1.0;
    TR(0,0) = 5.0;
    X_exact(0,0) = 5.5;
    int sign = 1;

    // Calculate op(TL)*X + ISGN*X*op(TR)
    gemm( trans_l, Op::NoTrans, one, TL, X_exact, zero, B );
    gemm( Op::NoTrans, trans_r, sign, X_exact, TR, one, B );


    T scale, xnorm;
    lasy2( Op::NoTrans, Op::NoTrans, 1, TL, TR, B, scale, X, xnorm );

    CHECK( X_exact(0,0) == Approx( scale * X(0,0) ) );

}

TEST_CASE( "2x2 sylvester solver gives correct result", "[utils]" ) {

    typedef float T;
    typedef std::size_t idx_t;
    typedef real_type<T> real_t;
    typedef std::complex<real_t> complex_t;

    using internal::colmajor_matrix;

    const T zero(0);
    const T one(1);
    const idx_t n1 = 2;
    const idx_t n2 = 2;
    const real_t eps = uroundoff<real_t>();

    std::unique_ptr<T[]> _TL(new T[n1*n1]);
    auto TL = colmajor_matrix<T>(&_TL[0], n1, n1);

    std::unique_ptr<T[]> _TR(new T[n2*n2]);
    auto TR = colmajor_matrix<T>(&_TR[0], n2, n2);

    std::unique_ptr<T[]> _B(new T[n1*n2]);
    auto B = colmajor_matrix<T>(&_B[0], n1, n2);

    std::unique_ptr<T[]> _X(new T[n1 * n2]);
    auto X = colmajor_matrix<T>(&_X[0], n1, n2);

    std::unique_ptr<T[]> _X_exact(new T[n1 * n2]);
    auto X_exact = colmajor_matrix<T>(&_X_exact[0], n1, n2);

    for( idx_t i = 0; i < n1; ++i )
        for( idx_t j = 0; j < n1; ++j )
            TL(i,j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for( idx_t i = 0; i < n2; ++i )
        for( idx_t j = 0; j < n2; ++j )
            TR(i,j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for( idx_t i = 0; i < n1; ++i )
        for( idx_t j = 0; j < n2; ++j )
            X_exact(i,j) = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    Op trans_l = Op::NoTrans;
    Op trans_r = Op::NoTrans;

    int sign = 1;

    // Calculate op(TL)*X + ISGN*X*op(TR)
    gemm( trans_l, Op::NoTrans, one, TL, X_exact, zero, B );
    gemm( Op::NoTrans, trans_r, sign, X_exact, TR, one, B );

    // Solve sylvester equation
    T scale, xnorm;
    lasy2( Op::NoTrans, Op::NoTrans, 1, TL, TR, B, scale, X, xnorm );

    // Check that X_exact == X
    for( idx_t i = 0; i < n1; ++i )
        for( idx_t j = 0; j < n2; ++j )
            CHECK( X_exact(i,j) == Approx( scale * X(i,j) ) );

}

