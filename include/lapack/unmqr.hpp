/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from `lapack::geqrf`
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __UNMQR_HH__
#define __UNMQR_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larft.hpp"

namespace lapack {

/** Multiplies the general m-by-n matrix C by Q from `lapack::geqrf` using a blocked code.
 * 
 * @param work Vector of size n*nb + nb*nb (m*nb + nb*nb if side == Side::Right).
 * @see unmqr( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template< typename TA, typename TC >
int unmqr(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    TA const* A, blas::idx_t lda,
    TA const* tau,
    TC* C, blas::idx_t ldc,
    blas::scalar_type<TA,TC>* work )
{
    using blas::max;
    using blas::min;
    using blas::internal::colmajor_matrix;

    // Constants
    const int nb = 32;      // number of blocks

    // Choose side 
    idx_t nQ, nw;
    if( side == Side::Left ){ nQ = m; nw = max( 1, n ); }
    else                    { nQ = n; nw = max( 1, m ); }

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( m < 0, -3 );
    lapack_error_if( n < 0, -4 );
    lapack_error_if( k < 0 || k > nQ, -5 );
    lapack_error_if( lda < nQ, -7 );
    lapack_error_if( ldc < m, -10 );

    // Quick return
    if (m == 0 || n == 0 || k == 0)
        return 0;

    // Matrix views
    auto _A = (side == Side::Left)
            ? colmajor_matrix<TA>( (TA*)A, m, k, lda )
            : colmajor_matrix<TA>( (TA*)A, n, k, lda );
    auto _C = colmajor_matrix<TC>( C, m, n, ldc );

    // Preparing loop indexes
    idx_t i0, iN, step;
    if( (side == Side::Left && trans != Op::NoTrans) ||
        (side != Side::Left && trans == Op::NoTrans) ) {
        i0 = 0;
        iN = k-1+nb;
        step = nb;
    }
    else {
        i0 = ( (k-1) / nb ) * nb;
        iN = -nb;
        step = -nb;
    }
    idx_t mi, ic, ni, jc;
    if( side == Side::Left ) { ni = n-1; jc = 0; }
    else                     { mi = m-1; ic = 0; }
    
    // Main loop
    for (idx_t i = i0; i != iN; i += step) {
        idx_t ib = min( nb, k-i );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        lapack::larft(  Direction::Forward, StoreV::Columnwise,
                        nQ-i, ib, &_A(i,i), lda, &tau[i], &work[nw*nb], nb );

        // H or H**H is applied to C[0:m-1,0:n-1]
        if( side == Side::Left ) { mi = m-i; ic = i; }
        else                     { ni = n-i; jc = i; }

        // Apply H or H**H
        lapack::larfb(  side, trans, Direction::Forward, StoreV::Columnwise,
                        mi, ni, ib, &_A(i,i), lda, &work[nw*nb], nb,
                        &_C(ic,jc), ldc, work );
    }

    return 0;
}

}

#endif // __UNMQR_HH__
