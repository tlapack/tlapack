/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
/// @copyright
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tlapack.hpp>
#include <tblas.hpp>

#include <new>
#include <chrono>   // for high_resolution_clock
#include <iostream>
#include <stdlib.h>
#include <mpreal.h> // multiprecision type mpfr::mpreal

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename TA>
inline void printMatrix(
    lapack::size_t m, lapack::size_t n,
    const TA *A, lapack::size_t lda )
{        
    for (lapack::size_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (lapack::size_t j = 0; j < n; ++j)
            std::cout << A[ i + j*lda ] << " ";
    }
}

//------------------------------------------------------------------------------
template <typename real_t>
void run( lapack::size_t m, lapack::size_t n )
{
    // Turn it off if m or n are large
    bool verbose = false;

    // Leading dimensions
    lapack::size_t lda = (m > 0) ? m : 1;
    lapack::size_t ldr = lda;
    lapack::size_t ldq = lda;

    // Arrays
    real_t* A = new real_t[ lda*n ];  // m-by-n
    real_t* Q = new real_t[ ldq*n ];  // m-by-n
    real_t* R = new real_t[ ldr*n ];  // m-by-n
    real_t* tau = new real_t[ n ];
    real_t* work;

    // Views
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define R(i_, j_) R[ (i_) + (j_)*ldr ]
    #define Q(i_, j_) Q[ (i_) + (j_)*ldq ]

    // Initialize arrays with junk
    for (lapack::size_t j = 0; j < n; ++j) {
        for (lapack::size_t i = 0; i < lda; ++i) { // lda == ldq == ldr
            A(i,j) = static_cast<float>( 0xDEADBEEF );
            Q(i,j) = static_cast<float>( 0xCAFED00D );
            R(i,j) = static_cast<float>( 0xFEE1DEAD );
        }
        tau[j] = static_cast<float>( 0xFFBADD11 );
    }
    
    // Generate a random matrix in A
    for (lapack::size_t j = 0; j < n; ++j)
        for (lapack::size_t i = 0; i < m; ++i)
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );

    // Frobenius norm of A
    real_t normA = lapack::lange(
        lapack::Norm::Fro,
        m, n, A, lda );

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(m,n,A,lda);
    }

    // Copy A to Q
    lapack::lacpy( lapack::Uplo::General, m, n, A, lda, Q, ldq );

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    
        // QR factorization
        blas_error_if( lapack::geqr2( m, n, Q, ldq, tau ) );

        // Save the R matrix
        lapack::lacpy( lapack::Uplo::Upper, n, n, Q, ldq, R, ldr );

        // Generates Q = H_1 H_2 ... H_n in the array A
        blas_error_if( lapack::org2r( m, n, n, Q, ldq, tau ) );
    
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR = std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);
    
    // Compute FLOPS
    double flopsQR = ( 4.0e+00 * ((double) m) * ((double) n) * ((double) n) 
                     - 4.0e+00 / 3.0e+00 * ((double) n) * ((double) n) * ((double) n)
                    ) / (elapsedQR.count() * 1.0e-9) ;

    // Print Q and R
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(m,n,Q,ldq);
        std::cout << std::endl << "R = ";
        printMatrix(m,n,R,ldr);
    }

    // 2) Compute ||Q'Q - I||_F

    work = new real_t[ n*n ];
    for (lapack::size_t i = 0; i < n*n; ++i) work[i] = static_cast<float>( 0xABADBABE );
        
        // work receives the identity n*n
        lapack::laset<real_t>(
            lapack::Layout::ColMajor, lapack::Uplo::Upper, n, n, 0.0, 1.0, work, n );
        // work receives Q'Q - I
        blas::syrk(
            lapack::Layout::ColMajor, lapack::Uplo::Upper,
            lapack::Op::Trans, n, m, 1.0, Q, ldq, -1.0, work, n );

        // Compute ||Q'Q - I||_F
        real_t norm_orth_1 = lapack::lansy( lapack::Norm::Fro, lapack::Uplo::Upper, n, work, n );

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(n,n,work,n);
        }

    delete[] work;

    // 3) Compute ||QR - A||_F / ||A||_F

    work = new real_t[ m*n ];
    for (lapack::size_t i = 0; i < n*n; ++i) work[i] = static_cast<float>( 0xABADBABE );

        // Copy Q to work
        lapack::lacpy( lapack::Uplo::General, m, n, Q, ldq, work, m );

        blas::trmm(
            blas::Layout::ColMajor, blas::Side::Right,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            m, n, 1.0, R, ldr, work, m );

        for(lapack::size_t j = 0; j < n; ++j)
            for(lapack::size_t i = 0; i < m; ++i)
                work[ i+j*m ] -= A(i,j);

        real_t norm_repres_1 = lapack::lange(
            lapack::Norm::Fro,
            m, n, work, m ) / normA;

    delete[] work;
    
    // *) Output
 
    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
            << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F  = " << norm_repres_1
            << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;

    delete[] A;
    delete[] R;
    delete[] Q;
    delete[] tau;

    #undef A
    #undef R
    #undef Q
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m, n;

    // Default arguments
    m = ( argc < 2 ) ? 7 : atoi( argv[1] );
    n = ( argc < 3 ) ? 5 : atoi( argv[2] );

    srand( 3 ); // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;
    
    printf( "run< float  >( %d, %d )", m, n );
    run< float  >( m, n );
    printf( "-----------------------\n" );
    
    printf( "run< double >( %d, %d )", m, n );
    run< double >( m, n );
    printf( "-----------------------\n" );
    
    printf( "run< long double >( %d, %d )", m, n );
    run< long double >( m, n );
    printf( "-----------------------\n" );

    return 0;
}
