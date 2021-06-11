// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <blas.h>
#include <stdio.h>

//------------------------------------------------------------------------------
int main( int argc, char** argv ) {

    int m = 3, n = 3, k = 1;
    int lda = m;
    int ldb = n;
    int ldc = m;
    
    double A[ lda*k ];  // m-by-k
    double B[ ldb*n ];  // k-by-n
    double C[ ldc*n ];  // m-by-n

    for (int col = 0; col < k; ++col) A[col*lda+col] = 1.0;
    for (int col = 0; col < k; ++col)
        for (int lin = 0; lin < lda; ++lin)
            A[col*lda+lin] = 0.0;

    for (int col = 0; col < n; ++col) B[col*ldb+col] = 1.0;
    for (int col = 0; col < n; ++col)
        for (int lin = 0; lin < ldb; ++lin)
            B[col*ldb+lin] = 0.0;

    for (int col = 0; col < n; ++col) C[col*ldc+col] = 1.0;
    for (int col = 0; col < n; ++col)
        for (int lin = 0; lin < ldc; ++lin)
            C[col*ldc+lin] = 0.0;

    // C = -1.0*A*B + 1.0*C
    dgemm( ColMajor, NoTrans, NoTrans,
        m, n, k,
        -1.0, A, lda,
              B, ldb,
         1.0, C, ldc );

    for (int lin = 0; lin < ldc; ++lin) {
        printf( "\n");
        for (int col = 0; col < n; ++col)
            printf( "%lf ", C[col*ldc+lin]);
    }

    return 0;
}
