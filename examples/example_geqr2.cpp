// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA

#include <tlapack.hpp>
#include <tblas.hpp>

#include <new>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>

template <typename TA>
inline void printMatrix(
    lapack::size_t m, lapack::size_t n,
    const TA *A, lapack::size_t lda ) {
        
    for (lapack::size_t i = 0; i < m; ++i) {
        printf( "\n" );
        for (lapack::size_t j = 0; j < n; ++j)
            printf( "%10.2e", A[ i + j*lda ] );
    }
}

//------------------------------------------------------------------------------
template <typename real_t>
void run( lapack::size_t m, lapack::size_t n )
{
    bool verbose = false;
    double elapsed_refL, perform_refL;
    struct timeval tp;

    lapack::size_t lda = (m > 0) ? m : 1;
    lapack::size_t ldr = lda;
    lapack::size_t ldq = lda;
    lapack::size_t tsize = std::min(m,n);

    real_t* A = new real_t[ lda*n ];  // m-by-n
    real_t* Q = new real_t[ lda*n ];  // m-by-n
    real_t* R = new real_t[ ldr*n ];  // m-by-n
    real_t* tau = new real_t[ tsize ];
    real_t* work;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define R(i_, j_) R[ (i_) + (j_)*ldr ]
    #define Q(i_, j_) Q[ (i_) + (j_)*ldq ]

    for (lapack::size_t j = 0; j < n; ++j)
        for (lapack::size_t i = 0; i < m; ++i)
            A(i,j) = static_cast<float>( rand() )
                        / static_cast<float>( RAND_MAX );
    real_t normA = lapack::lange(
        lapack::Norm::Fro,
        m, n, A, lda );

    if (verbose) {
        printf( "\nA = " );
        printMatrix(m,n,A,lda);
    }

    // Copy A to Q
    lapack::lacpy( lapack::Uplo::General, m, n, A, lda, Q, ldq );

    gettimeofday(&tp, NULL);
    elapsed_refL = -((double)tp.tv_sec+(1.e-6)*tp.tv_usec);

    // QR factorization
    blas_error_if( lapack::geqr2( m, n, Q, ldq, tau, tau+1 ) );
    // Save the R part
    lapack::lacpy( lapack::Uplo::Upper, m, n, Q, ldq, R, ldr );
    // Generates Q = H_1 H_2 ... H_n in the array A
    blas_error_if( lapack::org2r( m, n, n, Q, ldq, tau ) );

    gettimeofday(&tp, NULL);
    elapsed_refL += ((double)tp.tv_sec+(1.e-6)*tp.tv_usec);
    perform_refL = ( 4.0e+00 * ((double) m) * ((double) n) * ((double) n) 
                    - 4.0e+00 / 3.0e+00 * ((double) n) * ((double) n) * ((double) n)
                )  / elapsed_refL / 1.0e+9 ;

    if (verbose) {
        printf( "\nQ = " );
        printMatrix(m,n,Q,ldq);
        printf( "\nR = " );
        printMatrix(m,n,R,ldr);
    }

    // 1) Q'Q - I

    work = new real_t[ n*n ];
        
        // work receives the identity n*n
        lapack::laset<real_t>( lapack::Uplo::General, n, n, 0, 1, work, n );
        // work receives Q'Q - I
        blas::syrk(
            lapack::Layout::ColMajor, lapack::Uplo::Upper,
            lapack::Op::Trans, n, m, 1.0, Q, ldq, -1.0, work, n );

        // Compute ||Q'Q - I||_2
        real_t norm_orth_1 = lapack::lange( lapack::Norm::Fro, n, n, work, n );

        if (verbose) {
            printf( "\nQ'Q-I = " );
            printMatrix(n,n,work,n);
        }

    delete[] work;

    // 2) Q'Q - I

    work = new real_t[ m*n ];

        // Copy Q to work
        lapack::lacpy( lapack::Uplo::General, m, n, Q, ldq, work, m );

        blas::trmm(
            blas::Layout::ColMajor, blas::Side::Right,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            m, n, 1.0, R, ldr, work, m );

        for(lapack::size_t i = 0; i < m; ++i)
            for(lapack::size_t j = 0; j < n; ++j)
                work[ i+j*m ] -= A(i,j);

        real_t norm_repres_1 = lapack::lange(
            lapack::Norm::Fro,
            m, n, work, m ) / normA;

    delete[] work;
    
    // *) Output

    printf("\n");
    printf("time = %f,   GFlop/sec = %f", elapsed_refL, perform_refL);
    printf("\n");
    printf("||QR - A||_2/||A||_2  = %5.1e,        ||Q'Q - I||_2  = %5.1e",
        norm_repres_1, norm_orth_1);

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
    int m = 319;
    int n = 157;
    srand( (unsigned)time(NULL) );
    
    printf( "run< float  >( %d, %d )", m, n );
    run< float  >( m, n );
    printf( "\n" );
    
    printf( "run< double >( %d, %d )", m, n );
    run< double >( m, n );
    printf( "\n" );
    
    // printf( "\n\nrun< std::complex<float>  >( %d, %d )", m, n );
    // run< std::complex<float>  >( m, n );
    
    // printf( "\n\nrun< std::complex<double> >( %d, %d )", m, n );
    // run< std::complex<double> >( m, n );

    return 0;
}
