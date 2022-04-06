/// @file orm2r.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/ormr2.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
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
    class side_t, class trans_t >
int orm2r(
    side_t side, trans_t trans,
    matrixA_t& A,
    const tau_t& tau,
    matrixC_t& C,
    work_t& work )
{
    using idx_t = size_type< matrixA_t >;
    using T     = type_t< matrixA_t >;
    using pair  = std::pair<idx_t,idx_t>;

    // constants
    const T one( 1 );
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = size(tau);

    // check arguments
    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( access_denied( lowerTriangle, read_policy(A)  ), -3 );
    lapack_error_if( access_denied( band_t(0,0),   write_policy(A) ), -3 );

    // const expressions
    constexpr bool positiveInc = (
        ( (side == Side::Left) && !(trans == Op::Trans) ) ||
        ( !(side == Side::Left) && (trans == Op::Trans) )
    );
    constexpr idx_t i0 = (positiveInc) ? 0 : k-1;
    constexpr idx_t iN = (positiveInc) ? k :  -1;
    constexpr idx_t inc = (positiveInc) ? 1 : -1;

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    for (idx_t i = i0; i != iN; i += inc) {
        
        const auto& v = (side == Side::Left)
                      ? slice( A, pair{i,m}, i )
                      : slice( A, pair{i,n}, i );
        auto& Ci = (side == Side::Left)
                 ? rows( C, pair{i,m} )
                 : cols( C, pair{i,n} );
        
        const auto Aii = A(i,i);
        A(i,i) = one;
        larf( side, v, tau[i], Ci, work );
        A(i,i) = Aii;
    }

    return 0;
}

}

#endif // __ORM2R_HH__
