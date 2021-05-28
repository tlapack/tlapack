// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/geqr2.h
/// @author Rodney James, University of Colorado Denver, USA

#ifndef __GEQR2_HH__
#define __GEQR2_HH__

#include "blas/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace lapack {

/** Computes a QR factorization of a real matrix A.
 * 
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 * 
 * @param[in] m The number of rows of the matrix A.
 * @param[in] n The number of columns of the matrix A.
 * @param[in,out] A Real m-by-n matrix.
 *     On exit, the elements on and above the diagonal of the array
 *     contain the min(m,n)-by-n upper trapezoidal matrix R
 *     (R is upper triangular if m >= n); the elements below the diagonal,
 *     with the array tau, represent the unitary matrix Q as a
 *     product of elementary reflectors.
 * @param[in] lda
 * @param[in,out] tau Real vector of length min(m,n).
 *     The scalar factors of the elementary reflectors.
 * 
 * @ingroup geqrf
 */
template< typename TA >
void geqr2(
    blas::size_t m, blas::size_t n,
    TA* A, blas::size_t lda,
    real_type<TA>* tau )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const TA zero( 0.0 );
    const TA one( 1.0 );

    // check arguments
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( lda < m );

    // quick return
    if (n == 0) return;

	const size_t k = std::min( m, n-1 );
    for(size_t i = 0; i < k; ++i) {

        larfg( m-i, A(i,i), &(A(i+1,i)), 1, tau[i] );

        TA alpha = A(i,i);
        A(i,i) = one;
        larf( Side::Left, m-i, n-i-1, &(A(i,i)), 1, tau[i], A(i,i+1), lda, &(tau[i+1]) );
        A(i,i) = alpha;
	}
    if( n-1 < m )
        larfg( m-n+1, A(n-1,n-1), &(A(n,n-1)), 1, tau[n-1] );

    #undef A
}

} // lapack

#endif // __GEQR2_HH__