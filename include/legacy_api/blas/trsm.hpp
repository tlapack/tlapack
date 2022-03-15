// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_TRSM_HH
#define TBLAS_LEGACY_TRSM_HH

#include "blas/utils.hpp"
#include "blas/trsm.hpp"

namespace blas {

/**
 * Solve the triangular matrix-vector equation
 * \[
 *     op(A) X = \alpha B,
 * \]
 * or
 * \[
 *     X op(A) = \alpha B,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * X and B are m-by-n matrices, and A is an m-by-m or n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
 *
 * No test for singularity or near-singularity is included in this
 * routine. Such tests must be performed before calling this routine.
 * @see latrs for a more numerically robust implementation.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] side
 *     Whether $op(A)$ is on the left or right of X:
 *     - Side::Left:  $op(A) X = B$.
 *     - Side::Right: $X op(A) = B$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *
 * @param[in] trans
 *     The form of $op(A)$:
 *     - Op::NoTrans:   $op(A) = A$.
 *     - Op::Trans:     $op(A) = A^T$.
 *     - Op::ConjTrans: $op(A) = A^H$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] m
 *     Number of rows of matrices B and X. m >= 0.
 *
 * @param[in] n
 *     Number of columns of matrices B and X. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A is not accessed.
 *
 * @param[in] A
 *     - If side = Left:
 *       the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
 *     - If side = Right:
 *       the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A.
 *     - If side = left:  lda >= max(1, m).
 *     - If side = right: lda >= max(1, n).
 *
 * @param[in, out] B
 *     On entry,
 *     the m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
 *     On exit, overwritten by the solution matrix X.
 *
 * @param[in] ldb
 *     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
 *
 * @ingroup trsm
 */
template< typename TA, typename TB,
    disable_if_allow_optblas_t<TA,TB> = 0 >
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    blas::idx_t m,
    blas::idx_t n,
    blas::scalar_type<TA, TB> alpha,
    TA const *A, blas::idx_t lda,
    TB       *B, blas::idx_t ldb )
{    
    typedef blas::scalar_type<TA, TB> scalar_t;
    using blas::internal::colmajor_matrix;

    // constants
    const scalar_t zero( 0.0 );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( lda < ((side == Side::Left) ? m : n) );
    blas_error_if( ldb < ((layout == Layout::RowMajor) ? n : m) );

    // quick return
    if (m == 0 || n == 0)
        return;

    // adapt if row major
    if (layout == Layout::RowMajor) {
        side = (side == Side::Left)
            ? Side::Right
            : Side::Left;
        if (uplo == Uplo::Lower)
            uplo = Uplo::Upper;
        else if (uplo == Uplo::Upper)
            uplo = Uplo::Lower;
        std::swap( m , n );
    }

    // Matrix views
    const auto _A = (side == Side::Left)
                  ? colmajor_matrix<TA>( (TA*)A, m, m, lda )
                  : colmajor_matrix<TA>( (TA*)A, n, n, lda );
    auto _B = colmajor_matrix<TB>( B, m, n, ldb );

    // alpha == zero
    if (alpha == zero) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = 0; i < m; ++i)
                _B(i,j) = TB(0);
        }
        return;
    }

    // alpha != zero
    trsm( side, uplo, trans, diag, alpha, _A, _B );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_TRSM_HH
