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
#include "lapack/larfb.hpp"

namespace lapack {

/** Multiplies the general m-by-n matrix C by Q from `lapack::geqrf` using a blocked code.
 * 
 * @param work Vector of size n*nb + nb*nb (m*nb + nb*nb if side == Side::Right).
 * @see unmqr( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template<
    class matrixA_t, class matrixC_t,
    class tau_t, class matrixW_t,
    class side_t, class trans_t,
    enable_if_t<(
    /* Requires: */
    (
        is_same_v< side_t, left_side_t > || 
        is_same_v< side_t, right_side_t > 
    ) && (
        is_same_v< trans_t, noTranspose_t > || 
        is_same_v< trans_t, conjTranspose_t > ||
        is_same_v< trans_t, transpose_t >
    )
    ), int > = 0
>
int unmqr(
    side_t side, trans_t trans,
    const matrixA_t& A, const tau_t& tau,
    matrixC_t& C, matrixW_t& W )
{
    using idx_t = size_type< matrixC_t >;
    using pair  = std::pair<idx_t,idx_t>;
    using std::max;
    using std::min;

    // Constants
    const idx_t nb = 32; // number of blocks
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = nrows(A);
    const idx_t nw = ( is_same_v< side_t, left_side_t > ) ? max<idx_t>(1,n) : max<idx_t>(1,m);

    // Preparing loop indexes
    idx_t i0, iN, step;
    if(
        ( is_same_v< side_t, left_side_t > &&
        ! is_same_v< trans_t, noTranspose_t > )
    ||
        ( is_same_v< side_t, right_side_t > &&
          is_same_v< trans_t, noTranspose_t > )
    ){
        i0 = 0;
        iN = k-1+nb;
        step = nb;
    }
    else {
        i0 = ( (k-1) / nb ) * nb;
        iN = -nb;
        step = -nb;
    }
    
    // Main loop
    for (idx_t i = i0; i != iN; i += step) {
        
        idx_t ib = min( nb, k-i );
        const auto V = submatrix( A, pair(i,nA), pair(i,i+ib) );
        const auto taui = subvector( tau, pair(i,i+ib) );
        auto T = submatrix( W, pair(nw,nw+ib), pair(nw,nw+ib) );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        lapack::larft( forward, columnwise_storage, V, taui, T );

        // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
        auto Ci = ( is_same_v< side_t, left_side_t > )
           ? submatrix( C, pair(i,m), pair(0,n) )
           : submatrix( C, pair(0,m), pair(i,n) );

        // Apply H or H**H
        auto W0 = submatrix( W, pair(0,ib), pair(0,nw) );
        lapack::larfb(
            side, trans, forward, columnwise_storage,
            V, T, Ci, W0
        );
    }

    return 0;
}

}

#endif // __UNMQR_HH__
