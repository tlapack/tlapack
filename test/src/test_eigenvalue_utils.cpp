/// @file test_eigenvalue_utils.cpp
/// @brief Test eigenvalue utils from <T>LAPACK.
//
// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
// 

#include "legacy_api/blas/utils.hpp"
#include <catch2/catch.hpp>
#include <tlapack.hpp>

#include <complex>

using namespace blas;
using namespace lapack;

TEST_CASE( "eig22", "[eigenvalue_utils]" ) {


    const float eps = uroundoff<float>();
    float a11, a12, a21, a22;
    std::complex<float> s1, s2,s1_expected, s2_expected;

    // Handpicked example
    a11 = 1302;
    a12 = 801;
    a21 = -0.1;
    a22 = 6.8;

    s1_expected = std::complex<float>(1.301938153316081e3, 0.0);
    s2_expected = std::complex<float>(6.861846683919326, 0.0);

    lahqr_eig22(a11,a12,a21,a22, s1, s2);

    CHECK( abs1(s1 - s1_expected)/abs1(s1_expected)  < 1.0e4*eps );
    CHECK( abs1(s2 - s2_expected)/abs1(s2_expected)  < 1.0e4*eps );

    // Upper triangular example
    a11 = 0.4217613;
    a12 = 0.6557407;
    a21 = 0.0;
    a22 = 0.6787351;

    s1_expected = std::complex<float>(0.6787351, 0.0);
    s2_expected = std::complex<float>(0.4217613, 0.0);

    lahqr_eig22(a11,a12,a21,a22, s1, s2);

    CHECK( abs1(s1 - s1_expected)/abs1(s1_expected)  < 1.0e2*eps );
    CHECK( abs1(s2 - s2_expected)/abs1(s2_expected)  < 1.0e2*eps );

    // Lower triangular example
    a11 = 0.9157355;
    a12 = 0.0;
    a21 = 0.0357117;
    a22 = 0.7577401;

    s1_expected = std::complex<float>(0.9157355, 0.0);
    s2_expected = std::complex<float>(0.7577401, 0.0);

    lahqr_eig22(a11,a12,a21,a22, s1, s2);

    CHECK( abs1(s1 - s1_expected)/abs1(s1_expected)  < 1.0e2*eps );
    CHECK( abs1(s2 - s2_expected)/abs1(s2_expected)  < 1.0e2*eps );
}

TEST_CASE( "shiftcolumn", "[eigenvalue_utils]" ) {

    typedef float T;
    typedef float real_t;
    typedef std::complex<real_t> complex_t;
    using std::size_t;

    using blas::internal::colmajor_matrix;

    const T one(1);
    const T zero(0);
    const T eps = uroundoff<T>();
    const size_t n = 3;

    std::unique_ptr<T[]> _H(new T[n * n]);
    auto H = colmajor_matrix<T>(&_H[0], n, n, n);
    std::unique_ptr<T[]> _v(new T[n]);
    auto v = legacyVector<T>(n, &_v[0]);

    std::complex<real_t> s1,s2;

    s1 = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    s2 = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

    for( size_t i=0; i < n; ++i ){
        for( size_t j=0; j < n; ++j ){
            H( i,j ) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }

    lahqr_shiftcolumn( H, v, s1, s2 );

    std::unique_ptr<complex_t[]> _H1(new complex_t[n * n]);
    auto H1 = colmajor_matrix<complex_t>(&_H1[0], n, n, n);
    std::unique_ptr<complex_t[]> _H2(new complex_t[n * n]);
    auto H2 = colmajor_matrix<complex_t>(&_H2[0], n, n, n);
    std::unique_ptr<complex_t[]> _H3(new complex_t[n * n]);
    auto H3 = colmajor_matrix<complex_t>(&_H3[0], n, n, n);

    lacpy( Uplo::General, H, H1 );
    lacpy( Uplo::General, H, H2 );
    for( size_t i=0; i < n; ++i ){
        H1( i,i ) -= s1;
        H2( i,i ) -= s2;
    }
    gemm( Op::NoTrans, Op::NoTrans, one, H1, H2, zero, H3 );
    auto s =  ( abs1(v[0]) + abs1(v[1]) + abs1(v[2]) ) / ( abs1(H3(0,0)) + abs1(H3(1,0)) + abs1(H3(2,0)) )  ;

    CHECK( abs1( s*H3(0,0) - v[0] ) < 1.0e2*eps );
    CHECK( abs1( s*H3(1,0) - v[1] ) < 1.0e2*eps );
    CHECK( abs1( s*H3(2,0) - v[2] ) < 1.0e2*eps );

}
