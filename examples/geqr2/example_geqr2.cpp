/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <plugins/tlapack_mdspan.hpp>
#include <plugins/tlapack_stdvector.hpp>
#include <slate_api/blas/mdspan.hpp>
#include <tlapack.hpp>

#include <memory>
#include <vector>
#include <chrono>   // for high_resolution_clock
#include <iostream>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
inline void printMatrix( const matrix_t& A )
{
    using idx_t = blas::size_type< matrix_t >;
    const idx_t m = blas::nrows(A);
    const idx_t n = blas::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i,j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename real_t>
void run( size_t m, size_t n )
{
    using std::size_t;
    using blas::internal::colmajor_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Leading dimensions
    size_t lda = (m > 0) ? m : 1;
    size_t ldr = (n > 0) ? n : 1;
    size_t ldq = lda;

    // Arrays
    std::unique_ptr<real_t[]> _A(new real_t[ lda*n ]); // m-by-n
    std::unique_ptr<real_t[]> _R(new real_t[ ldr*n ]); // n-by-n
    std::unique_ptr<real_t[]> _Q(new real_t[ ldq*n ]); // m-by-n
    std::vector<real_t> tau ( n );

    // Matrix views
    auto A = colmajor_matrix<real_t>( &_A[0], m, n, lda );
    auto R = colmajor_matrix<real_t>( &_R[0], n, n, ldr );
    auto Q = colmajor_matrix<real_t>( &_Q[0], m, n, ldq );

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i,j) = static_cast<float>( 0xDEADBEEF );
            Q(i,j) = static_cast<float>( 0xCAFED00D );
        }
        for (size_t i = 0; i < n; ++i) {
            R(i,j) = static_cast<float>( 0xFEE1DEAD );
        }
        tau[j] = static_cast<float>( 0xFFBADD11 );
    }
    
    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );

    // Frobenius norm of A
    auto normA = lapack::lange( lapack::frob_norm, A );

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix( A );
    }

    // Copy A to Q
    lapack::lacpy( lapack::general_matrix, A, Q );

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now(); {
        std::vector<real_t> work( n-1 );
    
        // QR factorization
        blas_error_if( lapack::geqr2( Q, tau, work ) );

        // Save the R matrix
        lapack::lacpy( lapack::upper_triangle, Q, R );

        // Generates Q = H_1 H_2 ... H_n
        blas_error_if( lapack::org2r( n, Q, tau, work ) );
    }
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
        printMatrix( Q );
        std::cout << std::endl << "R = ";
        printMatrix( R );
    }

    real_t norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::unique_ptr<real_t[]> _work(new real_t[ n*n ]);
        auto work = colmajor_matrix<real_t>( &_work[0], n, n );
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i,j) = static_cast<float>( 0xABADBABE );
        
        // work receives the identity n*n
        lapack::laset( lapack::upper_triangle, 0.0, 1.0, work );
        // work receives Q'Q - I
        blas::syrk( blas::Uplo::Upper, blas::Op::Trans, 1.0, Q, -1.0, work );

        // Compute ||Q'Q - I||_F
        norm_orth_1 = lapack::lansy( lapack::frob_norm, lapack::upper_triangle, work );

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix( work );
        }

    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        std::unique_ptr<real_t[]> _work(new real_t[ m*n ]);
        auto work = colmajor_matrix<real_t>( &_work[0], m, n );
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work(i,j) = static_cast<float>( 0xABADBABE );

        // Copy Q to work
        lapack::lacpy( lapack::general_matrix, Q, work );

        blas::trmm( blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, 1.0, R, work );

        for(size_t j = 0; j < n; ++j)
            for(size_t i = 0; i < m; ++i)
                work(i,j) -= A(i,j);

        norm_repres_1 = lapack::lange( lapack::frob_norm, work ) / normA;
    }
    
    // *) Output
 
    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
            << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F  = " << norm_repres_1
            << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;
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
