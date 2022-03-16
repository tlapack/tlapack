/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from lapack::geqrf()
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
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

/** Multiplies the general m-by-n matrix C by Q from lapack::geqrf() using a blocked code.
 *
 * - side = Left,  trans = NoTrans:   $Q C$
 * - side = Right, trans = NoTrans:   $C Q$
 * - side = Left,  trans = ConjTrans: $Q^H C$
 * - side = Right, trans = ConjTrans: $C Q^H$
 *
 * where Q is a unitary matrix defined as the product of k
 * elementary reflectors, as returned by lapack::geqrf():
 * \[
 *     Q = H(0) H(1) \dots H(k).
 * \]
 *
 * Q is of order m if side = Left and of order n if side = Right.
 *
 * For real matrices, this is an alias for `lapack::ormqr`.
 * 
 * @tparam matrixA_t    A \<T\>LAPACK abstract matrix.
 * @tparam matrixC_t    A \<T\>LAPACK abstract matrix.
 * @tparam tau_t        A \<T\>LAPACK abstract vector.
 * @tparam side_t       Either left_side_t or right_side_t.
 * @tparam trans_t      Either noTranspose_t, transpose_t or conjTranspose_t.
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
 * @param[in] side
 *     - Side::Left:  apply $Q$ or $Q^H$ from the Left;
 *     - Side::Right: apply $Q$ or $Q^H$ from the Right.
 *
 * @param[in] trans
 *     - Op::NoTrans:   No transpose, apply $Q$;
 *     - Op::ConjTrans: Conjugate transpose, apply $Q^H$.
 *
 * @param[in] A
 *     - If side = Left,  the m-by-k matrix A;
 *     - if side = Right, the n-by-k matrix A.
 *     \n
 *     The i-th column must contain the vector which defines the
 *     elementary reflector H(i), for i = 0, 1, ..., k-1, as returned by
 *     geqrf() in the first k columns of its array argument A.
 *
 * @param[in] tau
 *     The vector tau of length k.
 *     tau[i] must contain the scalar factor of the elementary
 *     reflector H(i), as returned by geqrf().
 *
 * @param[in,out] C
 *     The m-by-n matrix C, stored in an ldc-by-n array.
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by
 *     $Q C$ or $Q^H C$ or $C Q^H$ or $C Q$.
 *
 * @param[in,out] opts Options.
 *      - opts.nb Block size.
 *      If opts.nb does not exist or opts.nb <= 0, nb assumes a default value.
 *      
 *      - opts.workPtr Workspace pointer.
 *          - Pointer to a matrix of size (nb)-by-(n+nb) if side == Side::Left.
 *          - Pointer to a matrix of size (nb)-by-(m+nb) if side == Side::Right.
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
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
    const matrixA_t& A, const tau_t& tau,
    matrixC_t& C, opts_t&& opts )
{
    using idx_t = size_type< matrixC_t >;
    using pair  = std::pair<idx_t,idx_t>;
    using std::max;
    using std::min;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = nrows(A);
    const idx_t nw = ( side == Side::Left ) ? max<idx_t>(1,n) : max<idx_t>(1,m);
    
    // Options
    const idx_t nb = get_nb(opts); // Block size
    auto W = get_work(opts); // (nb)-by-(nw+nb) matrix

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );

    // Preparing loop indexes
    idx_t i0, iN, step;
    if(
        ( (side == Side::Left) &&
          !(trans == Op::Trans) )
    ||
        ( (side == Side::Right) &&
          (trans == Op::Trans) )
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
        
        idx_t ib = min<idx_t>( nb, k-i );
        const auto V = submatrix( A, pair{i,nA}, pair{i,i+ib} );
        const auto taui = subvector( tau, pair{i,i+ib} );
        auto T = submatrix( W, pair{nw,nw+ib}, pair{nw,nw+ib} );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        larft( forward, columnwise_storage, V, taui, T );

        // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
        auto Ci = ( side == Side::Left )
           ? submatrix( C, pair{i,m}, pair{0,n} )
           : submatrix( C, pair{0,m}, pair{i,n} );

        // Apply H or H**H
        auto W0 = submatrix( W, pair{0,ib}, pair{0,nw} );
        larfb(
            side, trans, forward, columnwise_storage,
            V, T, Ci, W0
        );
    }

    return 0;
}

}

#endif // __UNMQR_HH__
