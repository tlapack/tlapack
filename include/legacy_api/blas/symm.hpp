// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TBLAS_LEGACY_SYMM_HH
#define TBLAS_LEGACY_SYMM_HH

#include "blas/utils.hpp"
#include "blas/symm.hpp"

namespace blas {

/**
 * Symmetric matrix-matrix multiply:
 * \[
 *     C = \alpha A B + \beta C,
 * \]
 * or
 * \[
 *     C = \alpha B A + \beta C,
 * \]
 * where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric matrix,
 * and B and C are m-by-n matrices.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] side
 *     The side the matrix A appears on:
 *     - Side::Left:  $C = \alpha A B + \beta C$,
 *     - Side::Right: $C = \alpha B A + \beta C$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced:
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] m
 *     Number of rows of the matrices B and C.
 *
 * @param[in] n
 *     Number of columns of the matrices B and C.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A and B are not accessed.
 *
 * @param[in] A
 *     - If side = Left:  The m-by-m matrix A, stored in an lda-by-m array.
 *     - If side = Right: The n-by-n matrix A, stored in an lda-by-n array.
 *
 * @param[in] lda
 *     Leading dimension of A.
 *     - If side = Left:  lda >= max(1, m).
 *     - If side = Right: lda >= max(1, n).
 *
 * @param[in] B
 *     The m-by-n matrix B, stored in an ldb-by-n array.
 *
 * @param[in] ldb
 *     Leading dimension of B. ldb >= max(1, n).
 *
 * @param[in] beta
 *     Scalar beta. If beta is zero, C need not be set on input.
 *
 * @param[in] C
 *     The m-by-n matrix C, stored in an lda-by-n array.
 *
 * @param[in] ldc
 *     Leading dimension of C. ldc >= max(1, n).
 *
 * @ingroup symm
 */
template< typename TA, typename TB, typename TC >
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::idx_t m, blas::idx_t n,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, blas::idx_t lda,
    TB const *B, blas::idx_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, blas::idx_t ldc )
{
    typedef blas::scalar_type<TA, TB, TC> scalar_t;
    using blas::internal::colmajor_matrix;

    // constants
    const scalar_t zero( 0.0 );
    const scalar_t one( 1.0 );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( lda < ((side == Side::Left) ? m : n) );
    blas_error_if( ldb < ((layout == Layout::RowMajor) ? n : m) );
    blas_error_if( ldc < ((layout == Layout::RowMajor) ? n : m) );

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
    const auto _B = colmajor_matrix<TB>( (TB*)B, m, n, ldb );
    auto _C = colmajor_matrix<TC>( C, m, n, ldc );

    // alpha == zero
    if (alpha == zero) {
        if (beta == zero) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    _C(i,j) = zero;
            }
        }
        else if (beta != one) {
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i < m; ++i)
                    _C(i,j) *= beta;
            }
        }
        return;
    }

    symm( side, uplo, alpha, _A, _B, beta, _C );
}

}  // namespace blas

#endif        //  #ifndef TBLAS_LEGACY_SYMM_HH
