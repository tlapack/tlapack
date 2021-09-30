// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRMM_HH
#define BLAS_TRMM_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Triangular matrix-matrix multiply:
 * \[
 *     B = \alpha op(A) B,
 * \]
 * or
 * \[
 *     B = \alpha B op(A),
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] side
 *     Whether $op(A)$ is on the left or right of B:
 *     - Side::Left:  $B = \alpha op(A) B$.
 *     - Side::Right: $B = \alpha B op(A)$.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero:
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *     - Uplo::General is illegal (see @ref gemm instead).
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
 *     Number of rows of matrix B. m >= 0.
 *
 * @param[in] n
 *     Number of columns of matrix B. n >= 0.
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
 *     The m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
 *
 * @param[in] ldb
 *     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
 *
 * @ingroup trmm
 */
template< typename TA, typename TB >
void trmm(
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

    // adapt if row major
    if (layout == Layout::RowMajor) {

        // check remaining arguments
        blas_error_if( ldb < n );

        side = (side == Side::Left)
            ? Side::Right
            : Side::Left;
        if (uplo == Uplo::Lower)
            uplo = Uplo::Upper;
        else if (uplo == Uplo::Upper)
            uplo = Uplo::Lower;
        std::swap( m , n );
    }
    else {
        // check remaining arguments
        blas_error_if( ldb < m );
    }

    // quick return
    if (m == 0 || n == 0)
        return;

    // Matrix views
    auto _A = (side == Side::Left)
            ? colmajor_matrix<const TA>( A, m, m, lda )
            : colmajor_matrix<const TA>( A, n, n, lda );
    auto _B = colmajor_matrix<TB>( B, m, n, ldb );

    // alpha == zero
    if (alpha == zero) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = 0; i < m; ++i)
                _B(i,j) = zero;
        }
        return;
    }

    // alpha != zero
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t k = 0; k < m; ++k) {
                        const scalar_t alphaBkj = alpha*_B(k,j);
                        for(idx_t i = 0; i < k; ++i)
                            _B(i,j) += _A(i,k)*alphaBkj;
                        _B(k,j) = (diag == Diag::NonUnit)
                                ? _A(k,k)*alphaBkj
                                : alphaBkj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t k = m-1; k != idx_t(-1); --k) {
                        const scalar_t alphaBkj = alpha*_B(k,j);
                        _B(k,j) = (diag == Diag::NonUnit)
                                ? _A(k,k)*alphaBkj
                                : alphaBkj;
                        for(idx_t i = k+1; i < m; ++i)
                            _B(i,j) += _A(i,k)*alphaBkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? _A(i,i)*_B(i,j)
                                    : _B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum += _A(k,i)*_B(k,j);
                        _B(i,j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? _A(i,i)*_B(i,j)
                                    : _B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum += _A(k,i)*_B(k,j);
                        _B(i,j) = alpha * sum;
                    }
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? conj(_A(i,i))*_B(i,j)
                                    : _B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum += conj(_A(k,i))*_B(k,j);
                        _B(i,j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? conj(_A(i,i))*_B(i,j)
                                    : _B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum += conj(_A(k,i))*_B(k,j);
                        _B(i,j) = alpha * sum;
                    }
                }
            }
        }
    }
    else { // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = n-1; j != idx_t(-1); --j) {

                    scalar_t alphaAkj = (diag == Diag::NonUnit)
                                    ? alpha*_A(j,j)
                                    : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,j) *= alphaAkj;

                    for(idx_t k = 0; k < j; ++k) {
                        alphaAkj = alpha*_A(k,j);
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAkj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {

                    scalar_t alphaAkj = (diag == Diag::NonUnit)
                                    ? alpha*_A(j,j)
                                    : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,j) *= alphaAkj;

                    for(idx_t k = j+1; k < n; ++k) {
                        alphaAkj = alpha*_A(k,j);
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t k = 0; k < n; ++k) {
                    for(idx_t j = 0; j < k; ++j) {
                        const scalar_t alphaAjk = alpha*_A(j,k);
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*_A(k,k)
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,k) *= alphaAkk;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    for(idx_t j = k+1; j < n; ++j) {
                        const scalar_t alphaAjk = alpha*_A(j,k);
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*_A(k,k)
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,k) *= alphaAkk;
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t k = 0; k < n; ++k) {
                    for(idx_t j = 0; j < k; ++j) {
                        const scalar_t alphaAjk = alpha*conj(_A(j,k));
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*conj(_A(k,k))
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,k) *= alphaAkk;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    for(idx_t j = k+1; j < n; ++j) {
                        const scalar_t alphaAjk = alpha*conj(_A(j,k));
                        for(idx_t i = 0; i < m; ++i)
                            _B(i,j) += _B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*conj(_A(k,k))
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        _B(i,k) *= alphaAkk;
                }
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMM_HH
