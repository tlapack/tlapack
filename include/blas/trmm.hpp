// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_TRMM_HH__
#define __TLAPACK_BLAS_TRMM_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * Triangular matrix-matrix multiply:
 * \[
 *     B := \alpha op(A) B,
 * \]
 * or
 * \[
 *     B := \alpha B op(A),
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
 * upper or lower triangular matrix.
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
 * @param[in] alpha Scalar.
 * @param[in] A
 *     - If side = Left: a m-by-m matrix.
 *     - If side = Right: a n-by-n matrix.
 * @param[in,out] B A m-by-n matrix.
 *
 * @ingroup trmm
 */
template< class matrixA_t, class matrixB_t, class alpha_t >
void trmm(
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    const alpha_t alpha,
    const matrixA_t& A,
    matrixB_t& B )
{    
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixA_t >;

    // using
    using scalar_t = scalar_type<alpha_t,TA,TB>;

    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    tblas_error_if( side != Side::Left &&
                   side != Side::Right );
    tblas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    tblas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    tblas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    tblas_error_if( nrows(A) != ncols(A) );
    tblas_error_if( nrows(A) != ((side == Side::Left) ? m : n) );

    tblas_error_if( access_denied( uplo, read_policy(A) ) );
    tblas_error_if( access_denied( dense, write_policy(B) ) );
    
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t k = 0; k < m; ++k) {
                        const auto alphaBkj = alpha*B(k,j);
                        for(idx_t i = 0; i < k; ++i)
                            B(i,j) += A(i,k)*alphaBkj;
                        B(k,j) = (diag == Diag::NonUnit)
                                ? A(k,k)*alphaBkj
                                : alphaBkj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t k = m-1; k != idx_t(-1); --k) {
                        const auto alphaBkj = alpha*B(k,j);
                        B(k,j) = (diag == Diag::NonUnit)
                                ? A(k,k)*alphaBkj
                                : alphaBkj;
                        for(idx_t i = k+1; i < m; ++i)
                            B(i,j) += A(i,k)*alphaBkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? A(i,i)*B(i,j)
                                    : B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum += A(k,i)*B(k,j);
                        B(i,j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? A(i,i)*B(i,j)
                                    : B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum += A(k,i)*B(k,j);
                        B(i,j) = alpha * sum;
                    }
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? conj(A(i,i))*B(i,j)
                                    : B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum += conj(A(k,i))*B(k,j);
                        B(i,j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                    ? conj(A(i,i))*B(i,j)
                                    : B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum += conj(A(k,i))*B(k,j);
                        B(i,j) = alpha * sum;
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
                                    ? alpha*A(j,j)
                                    : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alphaAkj;

                    for(idx_t k = 0; k < j; ++k) {
                        alphaAkj = alpha*A(k,j);
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAkj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {

                    scalar_t alphaAkj = (diag == Diag::NonUnit)
                                    ? alpha*A(j,j)
                                    : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alphaAkj;

                    for(idx_t k = j+1; k < n; ++k) {
                        alphaAkj = alpha*A(k,j);
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t k = 0; k < n; ++k) {
                    for(idx_t j = 0; j < k; ++j) {
                        const auto alphaAjk = alpha*A(j,k);
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*A(k,k)
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alphaAkk;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    for(idx_t j = k+1; j < n; ++j) {
                        const auto alphaAjk = alpha*A(j,k);
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*A(k,k)
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alphaAkk;
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t k = 0; k < n; ++k) {
                    for(idx_t j = 0; j < k; ++j) {
                        const auto alphaAjk = alpha*conj(A(j,k));
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*conj(A(k,k))
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alphaAkk;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    for(idx_t j = k+1; j < n; ++j) {
                        const auto alphaAjk = alpha*conj(A(j,k));
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) += B(i,k)*alphaAjk;
                    }

                    const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                            ? alpha*conj(A(k,k))
                                            : alpha;
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alphaAkk;
                }
            }
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_TRMM_HH__
