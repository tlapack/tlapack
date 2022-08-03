/// @file example_gemm.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack/legacy_api/blas.hpp>

#include <vector>
#include <iostream>
#include <chrono>   // for high_resolution_clock

#ifdef USE_MPFR
    #include <tlapack/plugins/mpreal.hpp>
#endif

//------------------------------------------------------------------------------
template <typename T>
void run( tlapack::idx_t m, tlapack::idx_t n, tlapack::idx_t k )
{
    using tlapack::idx_t;
    using tlapack::min;
    using tlapack::internal::colmajor_matrix;
    using tlapack::internal::rowmajor_matrix;
    
    // Column Major data
    idx_t lda = (m > 0) ? m : 1;
    idx_t ldb = (k > 0) ? k : 1;
    idx_t ldc = (m > 0) ? m : 1;
    std::vector<T> A_( lda*k, T(0) );    // m-by-k
    std::vector<T> B_( ldb*n, T(0) );    // k-by-n
    std::vector<T> C_( ldc*n, T(0) );    // m-by-n

    // Column Major Matrix views
    auto A = colmajor_matrix<T>( &A_[0], m, k, lda );
    auto B = colmajor_matrix<T>( &B_[0], k, n, ldb );
    auto C = colmajor_matrix<T>( &C_[0], m, n, ldc );

    // Row Major data
    std::vector<T> Ar_( m*k, T(0) );   // m-by-k
    std::vector<T> Br_( k*n, T(0) );   // k-by-n
    std::vector<T> Cr_( m*n, T(0) );   // m-by-n

    // Row Major Matrix views
    auto Ar = rowmajor_matrix( &Ar_[0], m, k );
    auto Br = rowmajor_matrix( &Br_[0], k, n );
    auto Cr = rowmajor_matrix( &Cr_[0], m, n );

    // Number of runs to measure the minimum execution time
    int Nruns = 10;
    std::chrono::nanoseconds bestTime;

    // Initialize A and Ar with junk
    for (idx_t j = 0; j < k; ++j)
        for (idx_t i = 0; i < m; ++i) {
            A(i,j)  = T( static_cast<float>( 0xDEADBEEF ) );
            Ar(i,j) = A(i,j);
        }
    
    // Generate a random matrix in a submatrix of A
    for (idx_t j = 0; j < min(k,n); ++j)
        for (idx_t i = 0; i < m; ++i) {
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );
            Ar(i,j) = A(i,j);
        }
                        

    // The diagonal of B is full of ones
    for (idx_t i = 0; i < min(k,n); ++i) {
        B(i,i) = 1.0;
        Br(i,i) = B(i,i);
    }

    // 1) Using legacy LAPACK interface:

    bestTime = std::chrono::nanoseconds::max();
    for( int run = 0; run < Nruns; ++run ) {

        // Set C using A
        for (idx_t j = 0; j < min(k,n); ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i,j) = A(i,j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

            // C = -1.0*A*B + 1.0*C
            tlapack::gemm( tlapack::Layout::ColMajor,
                        tlapack::Op::NoTrans,
                        tlapack::Op::NoTrans,
                        m, n, k,
                        T(-1.0), &A_[0], lda, &B_[0], ldb, 
                        T( 1.0), &C_[0], ldc );
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if( elapsed < bestTime ) bestTime = elapsed;
    }

    // Output
    std::cout << "Using legacy LAPACK interface:" << std::endl
              << "||C-AB||_F = " << tlapack::nrm2( n, &C_[0], 1 ) << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // Using abstract interface:

    bestTime = std::chrono::nanoseconds::max();
    for( int run = 0; run < Nruns; ++run ) {

        // Set C using A
        for (idx_t j = 0; j < min(k,n); ++j)
            for (idx_t i = 0; i < m; ++i)
                C(i,j) = A(i,j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

            // C = -1.0*A*B + 1.0*C
            tlapack::gemm( tlapack::Op::NoTrans,
                        tlapack::Op::NoTrans,
                        T(-1.0), A, B,
                        T( 1.0), C );
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if( elapsed < bestTime ) bestTime = elapsed;
    }

    // Output
    std::cout << "Using abstract interface:" << std::endl
              << "||C-AB||_F = " << tlapack::nrm2( n, &C_[0], 1 ) << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;

    // Using abstract interface with row major layout:

    bestTime = std::chrono::nanoseconds::max();
    for( int run = 0; run < Nruns; ++run ) {

        // Set C using A
        for (idx_t j = 0; j < min(k,n); ++j)
            for (idx_t i = 0; i < m; ++i)
                Cr(i,j) = Ar(i,j);

        // Record start time
        auto start = std::chrono::high_resolution_clock::now();

            // C = -1.0*A*B + 1.0*C
            tlapack::gemm( tlapack::Op::NoTrans,
                        tlapack::Op::NoTrans,
                        T(-1.0), Ar, Br,
                        T( 1.0), Cr );
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Compute elapsed time in nanoseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        // Update best time
        if( elapsed < bestTime ) bestTime = elapsed;
    }

    // Output
    std::cout << "Using abstract interface with row major layout:" << std::endl
              << "||C-AB||_F = " << tlapack::nrm2( n, &C_[0], 1 ) << std::endl
              << "time = " << bestTime.count() * 1.0e-6 << " ms" << std::endl;
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
