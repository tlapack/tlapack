/// @file unm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNM2R_HH
#define TLAPACK_UNM2R_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"

namespace tlapack {

template<
    class matrixA_t, class matrixC_t, class tau_t,
    class side_t, class trans_t >
inline constexpr
void unm2r_worksize(
    side_t side, trans_t trans,
    matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C, workinfo_t& workinfo,
    const workspace_opts_t<>& opts = {} )
{
    using idx_t = size_type< matrixA_t >;
    using pair = std::pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t nA = (side == Side::Left) ? m : n;

    auto v = slice( A, pair{0,nA}, 0 );
    larf_worksize( side, v, tau[0], C, workinfo, opts );
}

/** Applies unitary matrix Q to a matrix C.
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
 * @param work Vector of size n, if side = Side::Left, or m, if side = Side::Right.
 * 
 * @ingroup geqrf
 */
template<
    class matrixA_t, class matrixC_t, class tau_t,
    class side_t, class trans_t >
int unm2r(
    side_t side, trans_t trans,
    matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    const workspace_opts_t<>& opts = {} )
{
    using idx_t = size_type< matrixA_t >;
    using T     = type_t< matrixA_t >;
    using pair = std::pair<idx_t, idx_t>;

    // constants
    const T one( 1 );
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    const idx_t nA = (side == Side::Left) ? m : n;

    // check arguments
    tlapack_check_false( side != Side::Left &&
                     side != Side::Right );
    tlapack_check_false( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans );
    tlapack_check_false( trans == Op::Trans && is_complex<matrixA_t>::value );
    tlapack_check_false( access_denied( lowerTriangle, read_policy(A)  ) );
    tlapack_check_false( access_denied( band_t(0,0),   write_policy(A) ) );
    tlapack_check_false( access_denied( dense, write_policy(C) ) );

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]()
    {
        workinfo_t workinfo;
        unm2r_worksize( side, trans, A, tau, C, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo.size(), opts.work );
    }();
        
    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{ work };

    // const expressions
    const bool positiveInc = (
        ( (side == Side::Left) && !(trans == Op::NoTrans) ) ||
        (!(side == Side::Left) &&  (trans == Op::NoTrans) )
    );
    const idx_t i0 = (positiveInc) ? 0 : k-1;
    const idx_t iN = (positiveInc) ? k :  -1;
    const idx_t inc = (positiveInc) ? 1 : -1;

    // Main loop
    for (idx_t i = i0; i != iN; i += inc) {
        
        auto v = slice( A, pair{i,nA}, i );
        auto Ci = (side == Side::Left)
                 ? rows( C, pair{i,m} )
                 : cols( C, pair{i,n} );
        
        const auto Aii = A(i,i);
        A(i,i) = one;
        larf(
            side, v,
            (trans == Op::ConjTrans) ? conj(tau[i]) : tau[i],
            Ci, larfOpts );
        A(i,i) = Aii;
    }

    return 0;
}

}

#endif // TLAPACK_UNM2R_HH
