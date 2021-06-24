/// @file example_gemm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tblas.hpp>

#include <vector>
#include <iostream>
#include <chrono>   // for high_resolution_clock

#ifdef USE_MPFR
    #include <mpreal.h>
#endif

//------------------------------------------------------------------------------
template <typename T>
void run( blas::size_t m, blas::size_t n, blas::size_t k )
{
    using blas::size_t;
    using blas::min;

    size_t lda = (m > 0) ? m : 1;
    size_t ldb = (k > 0) ? k : 1;
    size_t ldc = (m > 0) ? m : 1;
    std::vector<T> A( lda*k, T(0) );    // m-by-k
    std::vector<T> B( ldb*n, T(0) );    // k-by-n
    std::vector<T> C( ldc*n, T(0) );    // m-by-n

    // Views
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define B(i_, j_) B[ (i_) + (j_)*ldb ]
    #define C(i_, j_) C[ (i_) + (j_)*ldc ]

    // Initialize A with junk
    for (size_t j = 0; j < k; ++j)
        for (size_t i = 0; i < m; ++i)
            A(i,j) = static_cast<float>( 0xDEADBEEF );
    
    // Generate a random matrix in a submatrix of A
    for (size_t j = 0; j < min(k,n); ++j)
        for (size_t i = 0; i < m; ++i)
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );

    // Set C using A
    for (size_t j = 0; j < min(k,n); ++j)
        for (size_t i = 0; i < m; ++i)
            C(i,j) = A(i,j);

    // The diagonal of B is full of ones
    for (size_t i = 0; i < min(k,n); ++i)
        B(i,i) = 1.0;

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

        // C = -1.0*A*B + 1.0*C
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::NoTrans,
                    blas::Op::NoTrans,
                    m, n, k,
                    T(-1.0), &A[0], lda,
                             &B[0], ldb,
                    T( 1.0), &C[0], ldc );
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Output
    std::cout << "||C-AB||_F = " << blas::nrm2( n, &C[0], 1 ) << std::endl
              << "time = " << elapsed.count() * 1.0e-6 << " ms" << std::endl;

    #undef A
    #undef B
    #undef C
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m, n, k;

    // Default arguments
    m = ( argc < 2 ) ? 100 : atoi( argv[1] );
    n = ( argc < 3 ) ? 200 : atoi( argv[2] );
    k = ( argc < 4 ) ?  50 : atoi( argv[3] );

    srand( 3 ); // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;
    
    printf( "run< float >( %d, %d, %d )\n", m, n, k );
    run< float  >( m, n, k );
    printf( "-----------------------\n" );

    printf( "run< double >( %d, %d, %d )\n", m, n, k );
    run< double >( m, n, k );
    printf( "-----------------------\n" );

    printf( "run< complex<float> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<float>  >( m, n, k );
    printf( "-----------------------\n" );

    printf( "run< complex<double> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<double> >( m, n, k );
    printf( "-----------------------\n" );

#ifdef USE_MPFR
    printf( "run< mpfr::mpreal >( %d, %d, %d )\n", m, n, k );
    run< mpfr::mpreal >( m, n, k );
    printf( "-----------------------\n" );

    printf( "run< complex<mpfr::mpreal> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<mpfr::mpreal> >( m, n, k );
    printf( "-----------------------\n" );
#endif

    return 0;
}
