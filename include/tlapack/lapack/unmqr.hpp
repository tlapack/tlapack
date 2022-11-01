/// @file unmqr.hpp Multiplies the general m-by-n matrix C by Q from tlapack::geqrf()
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNMQR_HH
#define TLAPACK_UNMQR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/larfb.hpp"

namespace tlapack {

/**
 * Options struct for unmqr
 */
template< class workT_t = void >
struct unmqr_opts_t : public workspace_opts_t<workT_t>
{
    inline constexpr unmqr_opts_t( const workspace_opts_t<workT_t>& opts = {} )
    : workspace_opts_t<workT_t>( opts ) {};
    
    size_type<workT_t> nb = 32; ///< Block size
};

/** Worspace query.
 * @see unmqr
 * 
 * @param[out] workinfo On return, contains the required workspace sizes.
 * 
 * @see unmqr
 */
template<
    class matrixA_t, class matrixC_t,
    class tau_t, class side_t, class trans_t,
    class workT_t = void
>
inline constexpr
void unmqr_worksize(
    side_t side, trans_t trans,
    const matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    workinfo_t& workinfo,
    const unmqr_opts_t<workT_t>& opts = {} )
{
    using idx_t     = size_type< matrixC_t >;
    using matrixT_t = deduce_work_t<
                        workT_t,
                        matrix_type< matrixA_t, tau_t >
                      >;
    using T         = type_t< matrixT_t >;
    using pair      = std::pair<idx_t, idx_t>;

    // Constants
    const idx_t k = size(tau);
    const idx_t nb = min<idx_t>( opts.nb, k );

    // Local workspace sizes
    workinfo.m = nb*sizeof(T);
    workinfo.n = nb;

    // larfb:
    workinfo_t workinfo2;
    {
        // Constants
        const idx_t m = nrows(C);
        const idx_t n = ncols(C);
        const idx_t nA = (side == Side::Left) ? m : n;

        // Functors
        Create<matrixT_t> new_matrix;
        
        // Empty matrices
        const auto V = slice( A, pair{0,nA}, pair{0,nb} );
        const auto matrixT = new_matrix( nullptr, nb, nb );

        // Internal workspace queries
        larfb_worksize( side, trans, forward, columnwise_storage, V, matrixT, C, workinfo2, opts );
    }
    
    // Additional workspace needed inside the routine
    workinfo.minMax( workinfo2 );
}

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
 * @param[in] opts Options.
 *      @c opts.work is used if whenever it has sufficient size.
 *      The sufficient size can be obtained through a workspace query.
 * 
 * @ingroup geqrf
 */
template<
    class matrixA_t, class matrixC_t,
    class tau_t, class side_t, class trans_t,
    class workT_t = void
>
int unmqr(
    side_t side, trans_t trans,
    const matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    const unmqr_opts_t<workT_t>& opts = {} )
{
    using idx_t     = size_type< matrixC_t >;
    using matrixT_t = deduce_work_t<
                        workT_t,
                        matrix_type< matrixA_t, tau_t >
                      >;
    
    using pair  = pair<idx_t,idx_t>;
    using std::max;
    using std::min;
    
    // Functor
    Create<matrixT_t> new_matrix;

    // Constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = (side == Side::Left) ? m : n;
    const idx_t nb = min<idx_t>( opts.nb, k );

    // check arguments
    tlapack_check_false( side != Side::Left &&
                         side != Side::Right );
    tlapack_check_false( trans != Op::NoTrans &&
                         trans != Op::Trans &&
                         trans != Op::ConjTrans );
    tlapack_check_false( trans == Op::Trans && is_complex<matrixA_t>::value );
    
    tlapack_check_false( access_denied( strictLower, read_policy(A) ) );
    tlapack_check_false( access_denied( dense, write_policy(C) ) );
    
    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]()
    {
        workinfo_t workinfo;
        unmqr_worksize( side, trans, A, tau, C, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo, opts.work );
    }();

    // Matrix T and recompute work
    auto matrixT = new_matrix( work, nb, nb, work );
    
    // Options to forward
    auto&& larfbOpts = workspace_opts_t<void>{ work };

    // Preparing loop indexes
    const bool positiveInc = (
        ( (side == Side::Left) && !(trans == Op::NoTrans) ) ||
        (!(side == Side::Left) &&  (trans == Op::NoTrans) )
    );
    const idx_t i0  = (positiveInc) ? 0     : ( (k-1) / nb ) * nb;
    const idx_t iN  = (positiveInc) ? ( (k-1) / nb + 1 ) * nb : -nb;
    const idx_t inc = (positiveInc) ? nb    : -nb;
    
    // Main loop
    for (idx_t i = i0; i != iN; i += inc) {
        
        idx_t ib = min<idx_t>( nb, k-i );
        const auto V = slice( A, pair{i,nA}, pair{i,i+ib} );
        const auto taui = slice( tau, pair{i,i+ib} );
        auto matrixTi   = slice( matrixT, pair{0,ib}, pair{0,ib} );

        // Form the triangular factor of the block reflector
        // $H = H(i) H(i+1) ... H(i+ib-1)$
        larft( forward, columnwise_storage, V, taui, matrixTi );

        // H or H**H is applied to either C[i:m,0:n] or C[0:m,i:n]
        auto Ci = ( side == Side::Left )
           ? slice( C, pair{i,m}, pair{0,n} )
           : slice( C, pair{0,m}, pair{i,n} );

        // Apply H or H**H
        larfb( side, trans, forward, columnwise_storage, V, matrixTi, Ci, larfbOpts );
    }

    return 0;
}

}

#endif // TLAPACK_UNMQR_HH
