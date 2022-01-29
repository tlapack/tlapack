// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYR2K_HH
#define BLAS_SYR2K_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Symmetric rank-k update:
 * \[
 *     C = \alpha A B^T + \alpha B A^T + \beta C,
 * \]
 * or
 * \[
 *     C = \alpha A^T B + \alpha B^T A + \beta C,
 * \]
 * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
 * and A and B are n-by-k or k-by-n matrices.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 *
 * @param[in] uplo
 *     What part of the matrix C is referenced,
 *     the opposite triangle being assumed from symmetry:
 *     - Uplo::Lower: only the lower triangular part of C is referenced.
 *     - Uplo::Upper: only the upper triangular part of C is referenced.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans: $C = \alpha A B^T + \alpha B A^T + \beta C$.
 *     - Op::Trans:   $C = \alpha A^T B + \alpha B^T A + \beta C$.
 *     - In the real    case, Op::ConjTrans is interpreted as Op::Trans.
 *       In the complex case, Op::ConjTrans is illegal (see @ref her2k instead).
 *
 * @param[in] n
 *     Number of rows and columns of the matrix C. n >= 0.
 *
 * @param[in] k
 *     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
 *     - Otherwise:          number of rows    of the matrix A. k >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A and B are not accessed.
 *
 * @param[in] A
 *     - If trans = NoTrans:
 *       the n-by-k matrix A, stored in an lda-by-k array [RowMajor: n-by-lda].
 *     - Otherwise:
 *       the k-by-n matrix A, stored in an lda-by-n array [RowMajor: k-by-lda].
 *
 * @param[in] lda
 *     Leading dimension of A.
 *     - If trans = NoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)],
 *     - Otherwise:          lda >= max(1, k) [RowMajor: lda >= max(1, n)].
 *
 * @param[in] B
 *     - If trans = NoTrans:
 *       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
 *     - Otherwise:
 *       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
 *
 * @param[in] ldb
 *     Leading dimension of B.
 *     - If trans = NoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)],
 *     - Otherwise:          ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
 *
 * @param[in] beta
 *     Scalar beta. If beta is zero, C need not be set on input.
 *
 * @param[in] C
 *     The n-by-n symmetric matrix C,
 *     stored in an lda-by-n array [RowMajor: n-by-lda].
 *
 * @param[in] ldc
 *     Leading dimension of C. ldc >= max(1, n).
 *
 * @ingroup syr2k
 */
template<
    class matrixA_t, class matrixB_t, class matrixC_t, 
    class alpha_t, class beta_t >
void syr2k(
    blas::Uplo uplo,
    blas::Op trans,
    const alpha_t& alpha, const matrixA_t& A, const matrixB_t& B,
    const beta_t& beta, matrixC_t& C )
{    
    // data traits
    using TA    = type_t< matrixA_t >;
    using TB    = type_t< matrixB_t >;
    using idx_t = size_type< matrixC_t >;

    // constants
    const idx_t n = (trans == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t k = (trans == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( nrows(B) != nrows(A) ||
                   ncols(B) != ncols(A) );
    blas_error_if( nrows(C) != ncols(C) ||
                   nrows(C) != n );

    if (trans == Op::NoTrans) {
        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = 0; i <= j; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaBjl = alpha*B(j,l);
                    auto alphaAjl = alpha*A(j,l);
                    for(idx_t i = 0; i <= j; ++i)
                        C(i,j) += A(i,l)*alphaBjl + B(i,l)*alphaAjl;
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {

                for(idx_t i = j; i < n; ++i)
                    C(i,j) *= beta;

                for(idx_t l = 0; l < k; ++l) {
                    auto alphaBjl = alpha*B(j,l);
                    auto alphaAjl = alpha*A(j,l);
                    for(idx_t i = j; i < n; ++i)
                        C(i,j) += A(i,l)*alphaBjl + B(i,l)*alphaAjl;
                }
            }
        }
    }
    else { // trans == Op::Trans
        using scalar_t = scalar_type<TA,TB>;

        if (uplo != Uplo::Lower) {
        // uplo == Uplo::Upper or uplo == Uplo::General
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = 0; i <= j; ++i) {
                    scalar_t sum1 = 0;
                    scalar_t sum2 = 0;
                    for(idx_t l = 0; l < k; ++l) {
                        sum1 += A(l,i) * B(l,j);
                        sum2 += B(l,i) * A(l,j);
                    }
                    C(i,j) = alpha*sum1 + alpha*sum2 + beta*C(i,j);
                }
            }
        }
        else { // uplo == Uplo::Lower
            for(idx_t j = 0; j < n; ++j) {
                for(idx_t i = j; i < n; ++i) {
                    scalar_t sum1 = 0;
                    scalar_t sum2 = 0;
                    for(idx_t l = 0; l < k; ++l) {
                        sum1 +=  A(l,i) * B(l,j);
                        sum2 +=  B(l,i) * A(l,j);
                    }
                    C(i,j) = alpha*sum1 + alpha*sum2 + beta*C(i,j);
                }
            }
        }
    }

    if (uplo == Uplo::General) {
        for(idx_t j = 0; j < n; ++j) {
            for(idx_t i = j+1; i < n; ++i)
                C(i,j) = C(j,i);
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMM_HH
