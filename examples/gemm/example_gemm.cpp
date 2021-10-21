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
void run( blas::idx_t m, blas::idx_t n, blas::idx_t k )
{
    // using blas::idx_t;
    // using blas::min;
    // using blas::internal::colmajor_matrix;
    // using blas::rowmajor_matrix;
    
    // // Column Major data
    // idx_t lda = (m > 0) ? m : 1;
    // idx_t ldb = (k > 0) ? k : 1;
    // idx_t ldc = (m > 0) ? m : 1;
    // std::vector<T> A( lda*k, T(0) );    // m-by-k
    // std::vector<T> B( ldb*n, T(0) );    // k-by-n
    // std::vector<T> C( ldc*n, T(0) );    // m-by-n

    // // Row Major data
    // idx_t ldar = (k > 0) ? k : 1;
    // idx_t ldbr = (n > 0) ? n : 1;
    // idx_t ldcr = (n > 0) ? n : 1;
    // std::vector<T> Ar( m*ldar, T(0) );   // m-by-k
    // std::vector<T> Br( k*ldbr, T(0) );   // k-by-n
    // std::vector<T> Cr( m*ldcr, T(0) );   // m-by-n

    // // Column Major Matrix views
    // auto _A = colmajor_matrix<T>( &A[0], m, k, lda );
    // auto _B = colmajor_matrix<T>( &B[0], k, n, ldb );
    // auto _C = colmajor_matrix<T>( &C[0], m, n, ldc );

    // // Row Major Matrix views
    // auto _Ar = rowmajor_matrix<T>( &Ar[0], m, k, ldar );
    // auto _Br = rowmajor_matrix<T>( &Br[0], k, n, ldbr );
    // auto _Cr = rowmajor_matrix<T>( &Cr[0], m, n, ldcr );

    // // Number of runs to measure the minimum execution time
    // int Nruns = 10;
    // std::chrono::nanoseconds bestTime;

    // // Initialize A and Ar with junk
    // for (idx_t j = 0; j < k; ++j)
    //     for (idx_t i = 0; i < m; ++i) {
    //         _A(i,j)  = T( static_cast<float>( 0xDEADBEEF ) );
    //         _Ar(i,j) = _A(i,j);
    //     }
    
    // // Generate a random matrix in a submatrix of A
    // for (idx_t j = 0; j < min(k,n); ++j)
    //     for (idx_t i = 0; i < m; ++i) {
    //         _A(i,j) = static_cast<float>( rand() )
    //                     / static_cast<float>( RAND_MAX );
    //         _Ar(i,j) = _A(i,j);
    //     }
                        

    // // The diagonal of B is full of ones
    // for (idx_t i = 0; i < min(k,n); ++i) {
    //     _B(i,i) = 1.0;
    //     _Br(i,i) = _B(i,i);
    // }

    // // 1) Using classical LAPACK interface:

    // bestTime = std::chrono::nanoseconds::max();
    // for( int run = 0; run < Nruns; ++run ) {

    //     // Set C using A
    //     for (idx_t j = 0; j < min(k,n); ++j)
    //         for (idx_t i = 0; i < m; ++i)
    //             _C(i,j) = _A(i,j);

    //     // Record start time
    //     auto start = std::chrono::high_resolution_clock::now();

    //         // C = -1.0*A*B + 1.0*C
    //         blas::gemm( blas::Layout::ColMajor,
    //                     blas::Op::NoTrans,
    //                     blas::Op::NoTrans,
    //                     m, n, k,
    //                     T(-1.0), &A[0], lda,
    //                              &B[0], ldb,
    //                     T( 1.0), &C[0], ldc );
        
    //     // Record end time
    //     auto end = std::chrono::high_resolution_clock::now();

    //     // Compute elapsed time in nanoseconds
    //     auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    //     // Update best time
    //     if( elapsed < bestTime ) bestTime = elapsed;
    // }

    // // Output
    // std::cout << "Using classical LAPACK interface:" << std::endl
    //           << "||C-AB||_F = " << blas::nrm2( n, &C[0], 1 ) << std::endl
    //           << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // // Using mdspan interface:

    // bestTime = std::chrono::nanoseconds::max();
    // for( int run = 0; run < Nruns; ++run ) {

    //     // Set C using A
    //     for (idx_t j = 0; j < min(k,n); ++j)
    //         for (idx_t i = 0; i < m; ++i)
    //             _C(i,j) = _A(i,j);

    //     // Record start time
    //     auto start = std::chrono::high_resolution_clock::now();

    //         // C = -1.0*A*B + 1.0*C
    //         blas::gemm( blas::Op::NoTrans,
    //                     blas::Op::NoTrans,
    //                     T(-1.0), _A, _B,
    //                     T( 1.0), _C );
        
    //     // Record end time
    //     auto end = std::chrono::high_resolution_clock::now();

    //     // Compute elapsed time in nanoseconds
    //     auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    //     // Update best time
    //     if( elapsed < bestTime ) bestTime = elapsed;
    // }

    // // Output
    // std::cout << "Using mdspan interface:" << std::endl
    //           << "||C-AB||_F = " << blas::nrm2( n, &C[0], 1 ) << std::endl
    //           << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // // Using mdspan interface with row major layout:

    // bestTime = std::chrono::nanoseconds::max();
    // for( int run = 0; run < Nruns; ++run ) {

    //     // Set C using A
    //     for (idx_t j = 0; j < min(k,n); ++j)
    //         for (idx_t i = 0; i < m; ++i)
    //             _Cr(i,j) = _Ar(i,j);

    //     // Record start time
    //     auto start = std::chrono::high_resolution_clock::now();

    //         // C = -1.0*A*B + 1.0*C
    //         blas::gemm( blas::Op::NoTrans,
    //                     blas::Op::NoTrans,
    //                     T(-1.0), _Ar, _Br,
    //                     T( 1.0), _Cr );
        
    //     // Record end time
    //     auto end = std::chrono::high_resolution_clock::now();

    //     // Compute elapsed time in nanoseconds
    //     auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    //     // Update best time
    //     if( elapsed < bestTime ) bestTime = elapsed;
    // }

    // // Output
    // std::cout << "Using mdspan interface with row major layout:" << std::endl
    //           << "||C-AB||_F = " << blas::nrm2( n, &Cr[0], 1 ) << std::endl
    //           << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;
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
