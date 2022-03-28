/// @file example_gehd2.cpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "legacy_api/blas/utils.hpp"
#include <plugins/tlapack_stdvector.hpp>
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
void run( size_t n )
{
    using std::size_t;
    using blas::internal::colmajor_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Leading dimensions
    size_t lda = (n > 0) ? n : 1;
    size_t ldh = (n > 0) ? n : 1;
    size_t ldq = lda;

    // Arrays
    std::unique_ptr<real_t[]> _A(new real_t[ lda*n ]); // m-by-n
    std::unique_ptr<real_t[]> _H(new real_t[ ldh*n ]); // n-by-n
    std::unique_ptr<real_t[]> _Q(new real_t[ ldq*n ]); // m-by-n
    std::vector<real_t> tau ( n );

    // Matrix views
    auto A = colmajor_matrix<real_t>( &_A[0], n, n, lda );
    auto H = colmajor_matrix<real_t>( &_H[0], n, n, ldh );
    auto Q = colmajor_matrix<real_t>( &_Q[0], n, n, ldq );

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            A(i,j) = static_cast<float>( 0xDEADBEEF );
            Q(i,j) = static_cast<float>( 0xCAFED00D );
        }
        for (size_t i = 0; i < n; ++i) {
            H(i,j) = static_cast<float>( 0xFEE1DEAD );
        }
        tau[j] = static_cast<float>( 0xFFBADD11 );
    }
    
    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );

    // for (size_t j = 0; j < n; ++j)
    //     for (size_t i = j+2; i < n; ++i)
    //         A(i,j) = 0.0;

    // Frobenius norm of A
    auto normA = lapack::lange( lapack::frob_norm, A );

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix( A );
    }

    // Copy A to Q
    lapack::lacpy( lapack::general_matrix, A, Q );

    // 1) Compute A = QHQ* (Stored in the matrix Q)

    // Record start time
    auto startQHQ = std::chrono::high_resolution_clock::now(); {
        std::vector<real_t> work( n-1 );
    
        // Hessenberg factorization
        blas_error_if( lapack::gehd2( 0, n, Q, tau, work ) );

        // Save the H matrix
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < std::min(n,j+2); ++i)
                H(i,j) = Q(i,j);

        // Generate Q = H_1 H_2 ... H_n
        blas_error_if( lapack::orghr( 0, n, Q, tau, work ) );

        // Remove junk from lower half of H
        for (size_t j = 0; j < n; ++j)
            for (size_t i = j+2; i < n; ++i)
                H(i,j) = 0.0;

        // Shur factorization
        std::vector<real_t> w( n );
        blas_error_if( lapack::lahqr( true, true, 0, n, H, w, Q ) );
    }
    // Record end time
    auto endQHQ = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQHQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endQHQ - startQHQ);

    // Print Q and H
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix( Q );
        std::cout << std::endl << "H = ";
        printMatrix( H );
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

    // 3) Compute ||QHQ* - A||_F / ||A||_F

    {
        std::unique_ptr<real_t[]> _work(new real_t[ n*n ]);
        auto work = colmajor_matrix<real_t>( &_work[0], n, n );
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
                work(i,j) = static_cast<float>( 0xABADBABC );

        blas::gemm( blas::Op::NoTrans, blas::Op::NoTrans, 1.0, Q, H, 0.0, work );
        blas::gemm( blas::Op::NoTrans, blas::Op::Trans, 1.0, work, Q, 0.0, H );

        for(size_t j = 0; j < n; ++j)
            for(size_t i = 0; i < n; ++i)
                H(i,j) -= A(i,j);

        norm_repres_1 = lapack::lange( lapack::frob_norm, H ) / normA;

    }
    
    // *) Output
 
    std::cout << std::endl;
    std::cout << "time = " << elapsedQHQ.count() * 1.0e-6 << " ms";
    std::cout << std::endl;
    std::cout << "||QHQ* - A||_F/||A||_F  = " << norm_repres_1
            << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int n;

    // Default arguments
    n = ( argc < 2 ) ? 5 : atoi( argv[1] );

    srand( 3 ); // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;
    
    printf( "run< float  >( %d )", n );
    run< float  >( n );
    printf( "-----------------------\n" );
    
    printf( "run< double >( %d )", n );
    run< double >( n );
    printf( "-----------------------\n" );
    
    printf( "run< long double >( %d )", n );
    run< long double >( n );
    printf( "-----------------------\n" );

    return 0;
}
