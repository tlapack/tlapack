/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from tlapack::geqrf()
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UNMQR_HH__
#define __TLAPACK_UNMQR_HH__

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larft.hpp"
#include "lapack/larfb.hpp"

namespace tlapack {

/** Applies orthogonal matrix op(Q) to a matrix C using a blocked code.
 *
 * - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 * - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 * - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 * - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
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
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * @tparam trans_t Either Op or any class that implements `operator Op()`. 
 * @tparam opts_t
 * \code{.cpp}
 *      struct opts_t {
 *          idx_t nb; // Block size
 *          matrix_t* workPtr; // Workspace pointer
 *          // ...
 *      };
 * \endcode
 *      If opts_t::nb does not exist, nb assumes a default value.
 *
 * @param[in] side Specifies which side op(Q) is to be applied.
 *      - Side::Left:  C := op(Q) C;
 *      - Side::Right: C := C op(Q).
 * 
 * @param[in] trans The operation $op(Q)$ to be used:
 *      - Op::NoTrans:      $op(Q) = Q$;
 *      - Op::ConjTrans:    $op(Q) = Q^H$.
 *      Op::Trans is a valid value if the data type of A is real. In this case,
 *      the algorithm treats Op::Trans as Op::ConjTrans.
 * 
 * @param[in] A
 *      - side = Side::Left:    m-by-k matrix;
 *      - side = Side::Right:   n-by-k matrix.
 * 
 * @param[in] tau Vector of length k
 *      Contains the scalar factors of the elementary reflectors.
 * 
 * @param[in,out] C m-by-n matrix. 
 *      On exit, C is replaced by one of the following:
 *      - side = Side::Left  & trans = Op::NoTrans:    $C := Q C$;
 *      - side = Side::Right & trans = Op::NoTrans:    $C := C Q$;
 *      - side = Side::Left  & trans = Op::ConjTrans:  $C := C Q^H$;
 *      - side = Side::Right & trans = Op::ConjTrans:  $C := Q^H C$.
 *
 * @param opts Options.
 *      - opts.nb [input] Block size.
 *      If opts.nb does not exist or opts.nb <= 0, nb assumes a default value.
 *      
 *      - opts.workPtr Workspace pointer.
 *          - Pointer to a matrix of size (nb)-by-(n+nb) if side = Side::Left.
 *          - Pointer to a matrix of size (nb)-by-(m+nb) if side = Side::Right.
 * 
 * @ingroup geqrf
 */
template<
    class matrixA_t, class matrixC_t,
    class tau_t, class side_t, class trans_t,
    class opts_t,
    enable_if_t< /// TODO: Remove this requirement when get_work() is fully functional
        has_work_v< opts_t >
    , int > = 0
>
int unmqr(
    side_t side, trans_t trans,
    const matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    opts_t&& opts )
{
    using idx_t = size_type< matrixC_t >;
    using pair  = pair<idx_t,idx_t>;
    using std::max;
    using std::min;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = (side == Side::Left) ? m : n;
    const idx_t nw = (side == Side::Left) ? max<idx_t>(1,n) : max<idx_t>(1,m);
    
    // Options
    const idx_t nb = get_nb(opts); // Block size
    auto W = get_work(opts); // (nb)-by-(nw+nb) matrix

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( trans == Op::Trans && is_complex<matrixA_t>::value, -2 );
    
    lapack_error_if( access_denied( strictLower, read_policy(A) ), -3 );
    lapack_error_if( access_denied( dense, write_policy(C) ), -5 );

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    // Preparing loop indexes
    const bool positiveInc = (
        ( (side == Side::Left) && !(trans == Op::NoTrans) ) ||
        (!(side == Side::Left) &&  (trans == Op::NoTrans) )
    );
    const idx_t i0 = (positiveInc) ? 0      : ( (k-1) / nb ) * nb;
    const idx_t iN = (positiveInc) ? k-1+nb : -nb;
    const idx_t inc = (positiveInc) ? nb    : -nb;
    
    // Main loop
    for (idx_t i = i0; i != iN; i += inc) {
        
        idx_t ib = min<idx_t>( nb, k-i );
        const auto V = slice( A, pair{i,nA}, pair{i,i+ib} );
        const auto taui = slice( tau, pair{i,i+ib} );
        auto T = slice( W, pair{nw,nw+ib}, pair{nw,nw+ib} );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        larft( forward, columnwise_storage, V, taui, T );

        // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
        auto Ci = ( side == Side::Left )
           ? slice( C, pair{i,m}, pair{0,n} )
           : slice( C, pair{0,m}, pair{i,n} );

        // Apply H or H**H
        auto W0 = slice( W, pair{0,ib}, pair{0,nw} );
        larfb(
            side, trans, forward, columnwise_storage,
            V, T, Ci, W0
        );
    }

    return 0;
}

}

#endif // __UNMQR_HH__
