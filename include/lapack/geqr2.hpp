/// @file geqr2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/geqr2.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __GEQR2_HH__
#define __GEQR2_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"

namespace lapack {

/** Computes a QR factorization of a matrix A.
 * 
 * @param work Vector of size n-1.
 * @see geqr2( blas::idx_t, blas::idx_t, TA*, blas::idx_t, TA* )
 * 
 * @ingroup geqrf
 */
template< typename TA >
int geqr2(
    blas::idx_t m, blas::idx_t n,
    TA* A, blas::idx_t lda,
    TA* tau,
    TA* work )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // Local parameters
    int info = 0;

    // constants
    const TA one( 1.0 );

    // check arguments
    lapack_error_if( m < 0, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( lda < m, -4 );

    // quick return
    if (n <= 0) return 0;

	const idx_t k = std::min<idx_t>( m, n-1 );
    for(idx_t i = 0; i < k; ++i) {

        larfg( m-i, A(i,i), &(A(i+1,i)), 1, tau[i] );

        TA alpha = A(i,i);
        A(i,i) = one;

        info = larf( Side::Left, m-i, n-i-1, &(A(i,i)), 1, tau[i], &(A(i,i+1)), lda, work+i );

        A(i,i) = alpha;
	}
    if( n-1 < m )
        larfg( m-n+1, A(n-1,n-1), &(A(n,n-1)), 1, tau[n-1] );

    #undef A

    return (info == 0) ? 0 : 1;
}

/** Computes a QR factorization of a complex matrix A.
 * 
 * @tparam real_t floating-point type.
 * Similar to @see geqr2( blas::idx_t, blas::idx_t, TA*, blas::idx_t, TA*, TA* )
 * but, here, A is complex and tau is real.
 * 
 * @note The imaginary part of tau is set to zero.
 * 
 * @ingroup geqrf
 */
template< typename real_t >
int geqr2(
    blas::idx_t m, blas::idx_t n,
    std::complex<real_t>* A, blas::idx_t lda,
    real_t* tau,
    std::complex<real_t>* work )
{
    typedef std::complex<real_t> scalar_t;
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t one( 1.0 );

    // check arguments
    lapack_error_if( m < 0, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( lda < m, -4 );

    // quick return
    if (n <= 0) return 0;

	const idx_t k = std::min<idx_t>( m, n-1 );
    for(idx_t i = 0; i < k; ++i) {

        larfg( m-i, A(i,i), &(A(i+1,i)), 1, tau[i] );

        scalar_t alpha = A(i,i);
        A(i,i) = one;

        larf( Side::Left, m-i, n-i-1, &(A(i,i)), 1, tau[i], &(A(i,i+1)), lda, work+i );

        A(i,i) = alpha;
	}
    if( n-1 < m )
        larfg( m-n+1, A(n-1,n-1), &(A(n,n-1)), 1, tau[n-1] );

    #undef A

    return 0;
}

/** Computes a QR factorization of a matrix A.
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
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * 
 * @param[in] m The number of rows of the matrix A.
 * @param[in] n The number of columns of the matrix A.
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 * @param[in] lda The leading dimension of A. lda >= max(1,m).
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *      The subarray tau[1:n-1] is used as workspace.
 * 
 * @ingroup geqrf
 */
template< typename TA >
inline int geqr2(
    blas::idx_t m, blas::idx_t n,
    TA* A, blas::idx_t lda,
    TA* tau )
{
    return geqr2( m, n, A, lda, tau, tau+1 );
}

/** Computes a QR factorization of a complex matrix A.
 * 
 * @tparam real_t floating-point type.
 * Similar to @see geqr2( blas::idx_t, blas::idx_t, TA*, blas::idx_t, TA* )
 * but, here, A is complex and tau is real.
 * 
 * @note The imaginary part of tau is set to zero.
 * 
 * @ingroup geqrf
 */
template< typename real_t >
int geqr2(
    blas::idx_t m, blas::idx_t n,
    std::complex<real_t>* A, blas::idx_t lda,
    real_t* tau )
{
    int info = 0;
    std::complex<real_t>* work
        = new std::complex<real_t>[ (n > 0) ? n-1 : 0 ];

    info = geqr2( m, n, A, lda, tau, work );

    delete[] work;
    return info;
}

} // lapack

#endif // __GEQR2_HH__