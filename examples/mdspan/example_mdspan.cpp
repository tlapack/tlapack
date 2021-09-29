/// @file example_mdspan.cpp Shows how to use mdspan to work with <T>LAPACK
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack.hpp>

#include <vector>
#include <iostream>

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using T = float;
    using namespace blas;
    
    // constants
    const int n = 100, k = 35, row_tile = 5, col_tile = 10;
    
    // raw data arrays
    T A_[ n*n ] = {};
    T C_[ n*n ] = {};

    /// Using dynamic extents: 

    // Column Major Matrix A
    auto A  = view_matrix( &A_[0], n, n, n );
    // Column Major Matrix Ak with the first k columns of A
    auto Ak = view_matrix( &A(0,0), n, k, n );
    // Tiled Matrix B with the last k*n elements of A_
    auto B  = Matrix<T,TiledLayout>(
        &A(n-k,0), TiledMapping( matrix_extents(k, n), row_tile, col_tile )
    );
    // Row Major Matrix C
    auto C  = Matrix<T,RowMajorLayout>(
        &C_[0], RowMajorMapping( matrix_extents(n, n), n )
    );

    // Init random seed
    srand( 4539 );

    // Initialize A with junk
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            A(i,j) = T( static_cast<float>( 0xDEADBEEF ) );
    
    // Generate a random matrix in Ak
    for (idx_t j = 0; j < k; ++j)
        for (idx_t i = 0; i < n; ++i)
            Ak(i,j) = static_cast<float>( rand() )
                    / static_cast<float>( RAND_MAX );
                        
    // Initialize B with zeros
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < k; ++i)
            B(i,j) = 0.0;
    // then put ones in the main diagonal of B
    for (idx_t i = 0; i < k; ++i)
        B(i,i) = 1.0;

    // Initialize the last n-k columns of C with zeros
    for (idx_t j = k; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            C(i,j) = 0.0;
    // Set the first k columns of C with Ak
    for (idx_t j = 0; j < k; ++j)
        for (idx_t i = 0; i < n; ++i)
            C(i,j) = Ak(i,j);

    // C = Ak*B + C
    gemm( Op::NoTrans, Op::NoTrans, T(-1.0), Ak, B, T(1.0), C );

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;
    std::cout << "|| C - Ak B ||_F = " 
              << lapack::lange( lapack::Norm::Fro, n, n, &C(0,0), n ) 
              << std::endl;

    return 0;
}
