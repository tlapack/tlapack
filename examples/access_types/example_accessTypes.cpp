/// @file example_access.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#undef TLAPACK_ERROR_NDEBUG
#undef NDEBUG

#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/lapack/lascl.hpp>

#include <vector>
#include <iostream>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
inline void printMatrix( const matrix_t& A )
{
    using idx_t = size_type< matrix_t >;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i,j) << " ";
    }
}

//------------------------------------------------------------------------------
/// Print banded matrix A in the standard output
template <typename matrix_t>
inline void printBandedMatrix( const matrix_t& A )
{
    using idx_t = size_type< matrix_t >;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = lowerband(A);
    const idx_t ku = upperband(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << (( i <= kl+j && j <= ku+i ) ? A(i,j) : 0 ) << " ";
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m = 5, n = 4;

    // Raw data 1:
    std::vector<int> data1(m*n);
    for (int i = 0; i < m*n; ++i)
        data1[i] = i+1;

    // Matrix 1
    legacyMatrix<int,size_t,Layout::RowMajor> A1( m, n, &data1[0], n );
    printMatrix(A1);
    std::cout << std::endl;

    // Raw data 2:
    std::vector<int> data2(m*n);
    for (int i = 0; i < m*n; ++i)
        data2[i] = data1[i];

    // Matrix 2
    legacyMatrix<int,size_t,Layout::RowMajor> A2( m, n, &data2[0], n );

    // Scale all matrix by 3
    lascl( dense, 1.0, 3.0, A1 );
    printMatrix(A1);
    std::cout << std::endl;

    // Scale upper triangle by 1/3
    lascl( upperTriangle, 3.0, 1.0, A1 );
    printMatrix(A1);
    std::cout << std::endl;

    // Scale strict lower triangle by 1/3
    lascl( strictLower, 3.0, 1.0, A1 );
    printMatrix(A1);
    std::cout << std::endl;

    // Test we return to the initial configuration
    bool goodResult = true;
    for (int i = 0; i < m*n; ++i)
        goodResult = goodResult && (data2[i] == data1[i]);

    std::cout << std::endl;
    std::cout << "Scaling well? " << (goodResult ? "True" : "False") << std::endl;

    // Matrix 3
    legacyBandedMatrix<int> A3( m, m, n/2, n-n/2-1, &data1[0] );
    printBandedMatrix(A3);
    std::cout << std::endl;

    try {
        lascl( dense, 1.0, 3.0, A3 );
        printBandedMatrix(A3);
        std::cout << std::endl;
    }
    catch( const tlapack::check_error& e ) {
        std::cout << std::endl;
        std::cout << "Generates access error as predicted" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    try {
        lascl( MatrixAccessPolicy::UpperHessenberg, 1.0, 3.0, A3 );
        printBandedMatrix(A3);
        std::cout << std::endl;
    }
    catch( const tlapack::check_error& e ) {
        std::cout << std::endl;
        std::cout << "Generates access error as predicted" << std::endl;
        std::cerr << e.what() << std::endl;
    }

    // Scale all matrix by 3
    lascl( band_t( n/2, n-n/2-1 ), 1.0, 3.0, A3 );
    printBandedMatrix(A3);
    std::cout << std::endl;

    // Scale lower band by 1/3
    lascl( band_t( n/2, 0 ), 3.0, 1.0, A3 );
    printBandedMatrix(A3);
    std::cout << std::endl;

    // Scale main diagonal by 3
    lascl( band_t( 0, 0 ), 1.0, 3.0, A3 );
    printBandedMatrix(A3);
    std::cout << std::endl;

    // Scale upper band by 1/3
    lascl( band_t( 0, n-n/2-1 ), 3.0, 1.0, A3 );
    printBandedMatrix(A3);
    std::cout << std::endl;

    // Test we return to the initial configuration
    bool goodResult2 = true;
    for (int i = 0; i < m*n; ++i)
        goodResult2 = goodResult2 && (data2[i] == data1[i]);

    std::cout << std::endl;
    std::cout << "Scaling well? " << (goodResult2 ? "True" : "False") << std::endl;

    return !(goodResult && goodResult2);
}
