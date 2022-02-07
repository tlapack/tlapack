/// @file example_cwrapper_gemm.c
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <tblas.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

//------------------------------------------------------------------------------
int main( int argc, char** argv ) {

    // Constants
    const BLAS_SIZE_T m = ( argc < 2 ) ? 100 : atoi( argv[1] );
    const BLAS_SIZE_T n = ( argc < 3 ) ? 200 : atoi( argv[2] );
    const BLAS_SIZE_T k = ( argc < 4 ) ?  50 : atoi( argv[3] );
    const BLAS_SIZE_T lda = (m > 0) ? m : 1;
    const BLAS_SIZE_T ldb = (k > 0) ? k : 1;
    const BLAS_SIZE_T ldc = (m > 0) ? m : 1;
    
    // Arrays
    double *A, *B, *C;

    // Time measurements
    clock_t start, end;
    double cpu_time_used;

    // Views
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define B(i_, j_) B[ (i_) + (j_)*ldb ]
    #define C(i_, j_) C[ (i_) + (j_)*ldc ]
    
    // Allocate A, B, and C, and initialize all entries with 0
    A = calloc( lda*k, sizeof(double) ); // m-by-k
    B = calloc( ldb*n, sizeof(double) ); // k-by-n
    C = calloc( ldc*n, sizeof(double) ); // m-by-n

    // Initialize random seed
    srand( 3 );

    // Initialize A with junk
    for (BLAS_SIZE_T j = 0; j < k; ++j)
        for (BLAS_SIZE_T i = 0; i < m; ++i)
            A(i,j) = (float) 0xDEADBEEF;
    
    // Generate a random matrix in a submatrix of A
    for (BLAS_SIZE_T j = 0; j < min(k,n); ++j)
        for (BLAS_SIZE_T i = 0; i < m; ++i)
            A(i,j) = ( (float) rand() ) / RAND_MAX;

    // Set C using A
    for (BLAS_SIZE_T j = 0; j < min(k,n); ++j)
        for (BLAS_SIZE_T i = 0; i < m; ++i)
            C(i,j) = A(i,j);

    // Set the main diagonal of B with ones
    for (BLAS_SIZE_T i = 0; i < min(k,n); ++i)
        B(i,i) = 1.0;

    // Record start time
    start = clock();

        // C = -1.0*A*B + 1.0*C
        dgemm( ColMajor, NoTrans, NoTrans,
               m, n, k,
               -1.0, A, lda,
                     B, ldb,
                1.0, C, ldc );
    
    // Record end time
    end = clock();

    // Compute elapsed time in miliseconds
    cpu_time_used = (1000 * (double) (end - start)) / CLOCKS_PER_SEC;

    // Output
    printf( "||C-AB||_F = %lf\n", dnrm2( n, C, 1 ) );
    printf( "time = %lf ms\n", cpu_time_used );

    // Deallocate A, B, C
    free(A);
    free(B);
    free(C);

    #undef A
    #undef B
    #undef C

    return 0;
}
