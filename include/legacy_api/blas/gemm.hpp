// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_GEMM_HH
#define TBLAS_LEGACY_GEMM_HH

#include "blas/utils.hpp"
#include "blas/gemm.hpp"

namespace blas {

/**
 * General matrix-matrix multiply:
 * \[
 *     C = \alpha op(A) \times op(B) + \beta C,
 * \]
 * where $op(X)$ is one of
 *     $op(X) = X$,
 *     $op(X) = X^T$, or
 *     $op(X) = X^H$,
 * alpha and beta are scalars, and A, B, and C are matrices, with
 * $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] transA
 *     The operation $op(A)$ to be used:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] transB
 *     The operation $op(B)$ to be used:
 *     - Op::NoTrans:   $op(B) = B$.
 *     - Op::Trans:     $op(B) = B^T$.
 *     - Op::ConjTrans: $op(B) = B^H$.
 *
 * @param[in] m
 *     Number of rows of the matrix C and $op(A)$. m >= 0.
 *
 * @param[in] n
 *     Number of columns of the matrix C and $op(B)$. n >= 0.
 *
 * @param[in] k
 *     Number of columns of $op(A)$ and rows of $op(B)$. k >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A and B are not accessed.
 *
 * @param[in] A
 *     - If transA = NoTrans:
 *       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
 *     - Otherwise:
 *       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A.
 *     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
 *     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
 *
 * @param[in] B
 *     - If transB = NoTrans:
 *       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
 *     - Otherwise:
 *       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
 *
 * @param[in] ldb
 *     Leading dimension of B.
 *     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
 *     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
 *
 * @param[in] beta
 *     Scalar beta. If beta is zero, C need not be set on input.
 *
 * @param[in] C
 *     The m-by-n matrix C, stored in an ldc-by-n array [RowMajor: m-by-ldc].
 *
 * @param[in] ldc
 *     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
 *
 * @ingroup gemm
 */
template< typename TA, typename TB, typename TC,
    disable_if_allow_optblas_t<TA, TB, TC> = 0 >
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, blas::idx_t lda,
    TB const *B, blas::idx_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, blas::idx_t ldc )
{
    using blas::internal::colmajor_matrix;

    // redirect if row major
    if (layout == Layout::RowMajor) {
        return gemm(
            Layout::ColMajor,
            transB,
            transA,
            n, m, k,
            alpha,
            B, ldb,
            A, lda,
            beta,
            C, ldc );
    }
    else {
        // check layout
        blas_error_if_msg( layout != Layout::ColMajor,
            "layout != Layout::ColMajor && layout != Layout::RowMajor" );
    }

    // check arguments
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );
    blas_error_if( lda < ((transA != Op::NoTrans) ? k : m) );
    blas_error_if( ldb < ((transB != Op::NoTrans) ? n : k) );
    blas_error_if( ldc < m );

    // quick return
    if (m == 0 || n == 0)
        return;

    // Matrix views
    const auto _A = (transA == Op::NoTrans)
            ? colmajor_matrix<TA>( (TA*)A, m, k, lda )
            : colmajor_matrix<TA>( (TA*)A, k, m, lda );
    const auto _B = (transB == Op::NoTrans)
            ? colmajor_matrix<TB>( (TB*)B, k, n, ldb )
            : colmajor_matrix<TB>( (TB*)B, n, k, ldb );
    auto _C = colmajor_matrix<TC>( C, m, n, ldc );

    gemm( transA, transB, alpha, _A, _B, beta, _C );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_GEMM_HH
