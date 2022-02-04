// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRMV_HH
#define BLAS_TRMV_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Triangular matrix-vector multiply:
 * \[
 *     x = op(A) x,
 * \]
 * where $op(A)$ is one of
 *     $op(A) = A$,
 *     $op(A) = A^T$, or
 *     $op(A) = A^H$,
 *     $op(A) = conj(A)$,
 * x is a vector,
 * and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed to be zero.
 *     - Uplo::Lower: A is lower triangular.
 *     - Uplo::Upper: A is upper triangular.
 *
 * @param[in] trans
 *     The operation to be performed:
 *     - Op::NoTrans:   $x = A   x$,
 *     - Op::Trans:     $x = A^T x$,
 *     - Op::ConjTrans: $x = A^H x$.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *                      The diagonal elements of A are not referenced.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A matrix.
 * @param[in,out] x vector.
 *
 * @ingroup trmv
 */
template< class matrixA_t, class vectorX_t >
void trmv(
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

    // using
    using scalar_t = scalar_type<TA,TX>;

    // constants
    const auto n = nrows(A);
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

    if (trans == Op::NoTrans) {
        // Form x := A*x
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                scalar_t tmp = x[j];
                for (idx_t i = 0; i < j; ++i)
                    x[i] += tmp * A(i,j);
                if (nonunit)
                    x[j] *= A(j,j);
            }
        }
        else {
            // lower
            for (idx_t j = n-1; j != idx_t(-1); --j) {
                // note: NOT skipping if x[j] is zero ...
                scalar_t tmp = x[j];
                for (idx_t i = n-1; i >= j+1; --i)
                    x[i] += tmp * A(i,j);
                if (nonunit)
                    x[j] *= A(j,j);
            }
        }
    }
    else if (trans == Op::Conj) {
        // Form x := A*x
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                scalar_t tmp = x[j];
                for (idx_t i = 0; i < j; ++i)
                    x[i] += tmp * conj( A(i,j) );
                if (nonunit)
                    x[j] *= conj( A(j,j) );
            }
        }
        else {
            // lower
            for (idx_t j = n-1; j != idx_t(-1); --j) {
                // note: NOT skipping if x[j] is zero ...
                scalar_t tmp = x[j];
                for (idx_t i = n-1; i >= j+1; --i)
                    x[i] += tmp * conj( A(i,j) );
                if (nonunit)
                    x[j] *= conj( A(j,j) );
            }
        }
    }
    else if (trans == Op::Trans) {
        // Form  x := A^T * x
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = n-1; j != idx_t(-1); --j) {
                scalar_t tmp = x[j];
                if (nonunit)
                    tmp *= A(j,j);
                for (idx_t i = j - 1; i != idx_t(-1); --i)
                    tmp += A(i,j) * x[i];
                x[j] = tmp;
            }
        }
        else {
            // lower
            for (idx_t j = 0; j < n; ++j) {
                scalar_t tmp = x[j];
                if (nonunit)
                    tmp *= A(j,j);
                for (idx_t i = j + 1; i < n; ++i)
                    tmp += A(i,j) * x[i];
                x[j] = tmp;
            }
        }
    }
    else {
        // Form x := A^H * x
        // same code as above A^T * x case, except add conj()
        if (uplo == Uplo::Upper) {
            // upper
            for (idx_t j = n-1; j != idx_t(-1); --j) {
                scalar_t tmp = x[j];
                if (nonunit)
                    tmp *= conj( A(j,j) );
                for (idx_t i = j - 1; i != idx_t(-1); --i)
                    tmp += conj( A(i,j) ) * x[i];
                x[j] = tmp;
            }
        }
        else {
            // lower
            for (idx_t j = 0; j < n; ++j) {
                scalar_t tmp = x[j];
                if (nonunit)
                    tmp *= conj( A(j,j) );
                for (idx_t i = j + 1; i < n; ++i)
                    tmp += conj( A(i,j) ) * x[i];
                x[j] = tmp;
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMV_HH
