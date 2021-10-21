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
    using blas::internal::colmajor_matrix;

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

    // Matrix views
    auto _A = colmajor_matrix<TA>( A, m, n, lda );

    const idx_t k = std::min<idx_t>( m, n-1 );
    for(idx_t i = 0; i < k; ++i) {

        larfg( m-i, _A(i,i), &(_A(i+1,i)), 1, tau[i] );

        TA alpha = _A(i,i);
        _A(i,i) = one;

        info = larf( Side::Left, m-i, n-i-1, &(_A(i,i)), 1, tau[i], &(_A(i,i+1)), lda, work+i );

        _A(i,i) = alpha;
	}
    if( n-1 < m )
        larfg( m-n+1, _A(n-1,n-1), &(_A(n,n-1)), 1, tau[n-1] );

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
    using blas::internal::colmajor_matrix;

    // constants
    const scalar_t one( 1.0 );

    // check arguments
    lapack_error_if( m < 0, -1 );
    lapack_error_if( n < 0, -2 );
    lapack_error_if( lda < m, -4 );

    // quick return
    if (n <= 0) return 0;

    // Matrix views
    auto _A = colmajor_matrix<real_t>( A, m, n, lda );

	const idx_t k = std::min<idx_t>( m, n-1 );
    for(idx_t i = 0; i < k; ++i) {

        larfg( m-i, _A(i,i), &(_A(i+1,i)), 1, tau[i] );

        scalar_t alpha = _A(i,i);
        _A(i,i) = one;

        larf( Side::Left, m-i, n-i-1, &(_A(i,i)), 1, tau[i], &(_A(i,i+1)), lda, work+i );

        _A(i,i) = alpha;
	}
    if( n-1 < m )
        larfg( m-n+1, _A(n-1,n-1), &(_A(n,n-1)), 1, tau[n-1] );

    return 0;
}

} // lapack

#endif // __GEQR2_HH__
