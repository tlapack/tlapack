// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRSM_HH
#define BLAS_TRSM_HH

#include "blas/utils.hpp"

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
template< typename TA, typename TB >
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    blas::idx_t m,
    blas::idx_t n,
    blas::scalar_type<TA, TB> alpha,
    TA const *A_, blas::idx_t lda,
    TB       *B_, blas::idx_t ldb )
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
    auto A = (side == Side::Left)
            ? colmajor_matrix<const TA>( A_, m, m, lda )
            : colmajor_matrix<const TA>( A_, n, n, lda );
    auto B = colmajor_matrix<TB>( B_, m, n, ldb );

    // alpha == zero
    if (alpha == zero) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = 0; i < m; ++i)
                B(i,j) = zero;
        }
        return;
    }

    // alpha != zero
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = m-1; k != idx_t(-1); --k) {
                        if (diag == Diag::NonUnit)
                            B(k,j) /= A(k,k);
                        for(idx_t i = 0; i < k; ++i)
                            B(i,j) -= A(i,k)*B(k,j);
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = 0; k < m; ++k) {
                        if (diag == Diag::NonUnit)
                            B(k,j) /= A(k,k);
                        for(idx_t i = k+1; i < m; ++i)
                            B(i,j) -= A(i,k)*B(k,j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum -= A(k,i)*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / A(i,i)
                            : sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum -= A(k,i)*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / A(i,i)
                            : sum;
                    }
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum -= conj(A(k,i))*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / conj(A(i,i))
                            : sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum -= conj(A(k,i))*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / conj(A(i,i))
                            : sum;
                    }
                }
            }
        }
    }
    else { // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = 0; k < j; ++k) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(k,j);
                    }
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) /= A(j,j);
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = n-1; j != idx_t(-1); --j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = j+1; k < n; ++k) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(k,j);
                    }
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) /= A(j,j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= A(k,k);
                    }
                    for(idx_t j = 0; j < k; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(j,k);
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= A(k,k);
                    }
                    for(idx_t j = k+1; j < n; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(j,k);
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= conj(A(k,k));
                    }
                    for(idx_t j = 0; j < k; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*conj(A(j,k));
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= conj(A(k,k));
                    }
                    for(idx_t j = k+1; j < n; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*conj(A(j,k));
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
        }
    }
}

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
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A is not accessed.
 *
 * @param[in] A
 *     - If side = Left: the m-by-m matrix A.
 *     - If side = Right: the n-by-n matrix A.
 *
 * @param[in, out] B
 *     On entry, the m-by-n matrix B.
 *     On exit, overwritten by the solution matrix X.
 *
 * @ingroup trsm
 */
template< typename TA, typename TB, typename LA, typename LB >
void trsm(
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    blas::scalar_type<TA, TB> alpha,
    const Matrix<TA,LA>& A,
    Matrix<TB,LB>&       B )
{    
    typedef blas::scalar_type<TA, TB> scalar_t;

    // constants
    const scalar_t zero( 0.0 );
    const auto& m = B.extent(0);
    const auto& n = B.extent(1);

    // check arguments
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( A.extent(0) != A.extent(1) );
    blas_error_if(
        A.extent(0) != ((side == Side::Left) ? B.extent(0) : B.extent(1)) );

    // quick return
    if (m == 0 || n == 0)
        return;

    // alpha == zero
    if (alpha == zero) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = 0; i < m; ++i)
                B(i,j) = zero;
        }
        return;
    }

    // alpha != zero
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = m-1; k != idx_t(-1); --k) {
                        if (diag == Diag::NonUnit)
                            B(k,j) /= A(k,k);
                        for(idx_t i = 0; i < k; ++i)
                            B(i,j) -= A(i,k)*B(k,j);
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = 0; k < m; ++k) {
                        if (diag == Diag::NonUnit)
                            B(k,j) /= A(k,k);
                        for(idx_t i = k+1; i < m; ++i)
                            B(i,j) -= A(i,k)*B(k,j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum -= A(k,i)*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / A(i,i)
                            : sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum -= A(k,i)*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / A(i,i)
                            : sum;
                    }
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = 0; k < i; ++k)
                            sum -= conj(A(k,i))*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / conj(A(i,i))
                            : sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = m-1; i != idx_t(-1); --i) {
                        scalar_t sum = alpha*B(i,j);
                        for(idx_t k = i+1; k < m; ++k)
                            sum -= conj(A(k,i))*B(k,j);
                        B(i,j) = (diag == Diag::NonUnit)
                            ? sum / conj(A(i,i))
                            : sum;
                    }
                }
            }
        }
    }
    else { // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for(idx_t j = 0; j < n; ++j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = 0; k < j; ++k) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(k,j);
                    }
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) /= A(j,j);
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t j = n-1; j != idx_t(-1); --j) {
                    for(idx_t i = 0; i < m; ++i)
                        B(i,j) *= alpha;
                    for(idx_t k = j+1; k < n; ++k) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(k,j);
                    }
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) /= A(j,j);
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= A(k,k);
                    }
                    for(idx_t j = 0; j < k; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(j,k);
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= A(k,k);
                    }
                    for(idx_t j = k+1; j < n; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*A(j,k);
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for(idx_t k = n-1; k != idx_t(-1); --k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= conj(A(k,k));
                    }
                    for(idx_t j = 0; j < k; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*conj(A(j,k));
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
            else { // uplo == Uplo::Lower
                for(idx_t k = 0; k < n; ++k) {
                    if (diag == Diag::NonUnit) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,k) /= conj(A(k,k));
                    }
                    for(idx_t j = k+1; j < n; ++j) {
                        for(idx_t i = 0; i < m; ++i)
                            B(i,j) -= B(i,k)*conj(A(j,k));
                    }
                    for(idx_t i = 0; i < m; ++i)
                        B(i,k) *= alpha;
                }
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRSM_HH
