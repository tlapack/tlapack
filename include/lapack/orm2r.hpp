/// @file orm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __ORM2R_HH__
#define __ORM2R_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larf.hpp"

namespace lapack {

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @param work Vector of size n (m if side == Side::Right).
 * @see orm2r( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template<
    class matrixA_t, class matrixC_t, class tau_t, class work_t,
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
int orm2r(
    side_t side, trans_t trans,
    const matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    work_t& work )
{
    using idx_t = size_type< matrixA_t >;
    using T     = type_t< matrixA_t >;
    using blas::full_extent;

    // constants
    const T one( 1 );
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);
    constexpr bool leftSide = is_same_v< side_t, left_side_t >;
    constexpr bool positiveInc = (
        ( leftSide && !is_same_v< trans_t, noTranspose_t > ) ||
        ( !leftSide && is_same_v< trans_t, noTranspose_t > )
    );
    constexpr idx_t i0 = (positiveInc) ? 0 : k-1;
    constexpr idx_t iN = (positiveInc) ? k :  -1;
    constexpr idx_t inc = (positiveInc) ? 1 : -1;

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    for (idx_t i = i0; i != iN; i += inc) {
        
        const auto& v = (leftSide)
                      ? col( A, i, pair(i,m) )
                      : col( A, i, pair(i,n) );
        auto& Ci = (leftSide)
                 ? submatrix( C, pair(i,m), full_extent )
                 : submatrix( C, full_extent, pair(i,n) );
        
        const auto Aii = A(i,i);
        A(i,i) = one;
        larf( side, v, tau(i), Ci, work );
        A(i,i) = Aii;
    }

    return 0;
}

}

#endif // __ORM2R_HH__