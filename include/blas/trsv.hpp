// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRSV_HH
#define BLAS_TRSV_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Solve the triangular matrix-vector equation
 * \[
 *     op(A) x = b,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 * x and b are vectors,
 * and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
 *
 * No test for singularity or near-singularity is included in this
 * routine. Such tests must be performed before calling this routine.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero.
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *
 * @param[in] trans
 *     The equation to be solved:
 *     - Op::NoTrans:   $A   x = b$,
 *     - Op::Trans:     $A^T x = b$,
 *     - Op::ConjTrans: $A^H x = b$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *                      The diagonal elements of A are not referenced.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A     A n-by-n matrix.
 * @param[in,out] x
 *      On entry, the n-element vector b.
 *      On exit,  the n-element vector x.
 *
 * @ingroup trsv
 */
template< class matrixA_t, class vectorX_t >
void trsv(
    Uplo uplo,
    Op trans,
    Diag diag,
    const matrixA_t& A,
    vectorX_t& x )
{
    // data traits
    using TA    = type_t< matrixA_t >;
    using TX    = type_t< vectorX_t >;
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = nrows(A);
    const bool nonunit = (diag == Diag::NonUnit);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans &&
                   trans != Op::Conj );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( nrows(A) != ncols(A) );
    blas_error_if( size(x) != n );

    blas_error_if( access_denied( uplo, read_policy(A) ) );

    if (trans == Op::NoTrans) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = n - 1; j != idx_t(-1); --j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                if (nonunit) {
                    x[j] /= A(j,j);
                }
                auto tmp = x[j];
                for (idx_t i = j - 1; i != idx_t(-1); --i) {
                    x[i] -= tmp * A(i,j);
                }
            }
        }
        else {
            // lower
            for (idx_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero ...
                if (nonunit) {
                    x[j] /= A(j,j);
                }
                auto tmp = x[j];
                for (idx_t i = j + 1; i < n; ++i) {
                    x[i] -= tmp * A(i,j);
                }
            }
        }
    }
    else if (trans == Op::Conj) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = n - 1; j != idx_t(-1); --j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                if (nonunit) {
                    x[j] /= conj( A(j,j) );
                }
                auto tmp = x[j];
                for (idx_t i = j - 1; i != idx_t(-1); --i) {
                    x[i] -= tmp * conj( A(i,j) );
                }
            }
        }
        else {
            // lower
            for (idx_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero ...
                if (nonunit) {
                    x[j] /= conj( A(j,j) );
                }
                auto tmp = x[j];
                for (idx_t i = j + 1; i < n; ++i) {
                    x[i] -= tmp * conj( A(i,j) );
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // Form  x := A^{-T} * x
        
        using scalar_t = scalar_type<TA,TX>;
        
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = 0; j < n; ++j) {
                scalar_t tmp = x[j];
                for (idx_t i = 0; i < j; ++i) {
                    tmp -= A(i,j) * x[i];
                }
                if (nonunit) {
                    tmp /= A(j,j);
                }
                x[j] = tmp;
            }
        }
        else {
            // lower
            for (idx_t j = n - 1; j != idx_t(-1); --j) {
                scalar_t tmp = x[j];
                for (idx_t i = j + 1; i < n; ++i) {
                    tmp -= A(i,j) * x[i];
                }
                if (nonunit) {
                    tmp /= A(j,j);
                }
                x[j] = tmp;
            }
        }
    }
    else {
        // Form x := A^{-H} * x
        // same code as above A^{-T} * x case, except add conj()
        
        using scalar_t = scalar_type<TA,TX>;
        
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = 0; j < n; ++j) {
                scalar_t tmp = x[j];
                for (idx_t i = 0; i < j; ++i) {
                    tmp -= conj( A(i,j) ) * x[i];
                }
                if (nonunit) {
                    tmp /= conj( A(j,j) );
                }
                x[j] = tmp;
            }
        }
        else {
            // lower
            for (idx_t j = n - 1; j != idx_t(-1); --j) {
                scalar_t tmp = x[j];
                for (idx_t i = j + 1; i < n; ++i) {
                    tmp -= conj( A(i,j) ) * x[i];
                }
                if (nonunit) {
                    tmp /= conj( A(j,j) );
                }
                x[j] = tmp;
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRSV_HH
