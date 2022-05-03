/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LARF_HH__
#define __TLAPACK_LARF_HH__

#include "base/utils.hpp"

#include "tblas.hpp"

namespace tlapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - \tau v v^H.
 * \]
 * If tau = 0, then H is taken to be the unit matrix.
 * 
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * 
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 * 
 * @param[in] v Vector of size m if side = Side::Left,
 *                          or n if side = Side::Right.
 * 
 * @param[in] tau Value of tau in the representation of H.
 * 
 * @param[in,out] C
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by $H C$ if side = Side::Left,
 *                               or $C H$ if side = Side::Right.
 * 
 * @param work Workspace vector with length n if side = Side::Left,
 *                                       or m if side = Side::Right.
 * 
 * @ingroup auxiliary
 */
template< class side_t, class vector_t, class tau_t, class matrix_t, class work_t >
inline void larf(
    side_t side,
    vector_t const& v, const tau_t& tau,
    matrix_t& C, work_t& work )
{

    // data traits
    using T = type_t<matrix_t>;
    using idx_t = size_type< matrix_t >;
    using pair = pair<size_t,size_t>;

    // constants
    const T one(1.0);
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    // check arguments
    tlapack_error_if( side != Side::Left &&
                   side != Side::Right );
    tlapack_error_if(  access_denied( dense, write_policy(C) ) );

    // The following code was changed from:
    //
    // if( side == Side::Left ) {
    //     gemv(Op::NoTrans, one, C, v, zero, work);
    //     ger(-tau, work, v, C);
    // }
    // else{
    //     gemv(Op::ConjTrans, one, C, v, zero, work);
    //     ger(-tau, v, work, C);
    // }
    //
    // This is so that v[0] doesn't need to be changed to 1,
    // which is better for thread safety.

    if( side == Side::Left ) {
        auto w = slice(work,pair{0,n});
        copy( row(C, 0), w );
        for (idx_t i = 0; i < n; ++i )
            w[i] = conj(w[i]);
        if(m > 1){
            auto x = slice(v,pair{1,m});
            gemv(Op::ConjTrans, one, rows(C, pair{1,m}), x, one, w);
        }
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = -tau * conj( w[j] );
            C(0,j) += tmp;
            for (idx_t i = 1; i < m; ++i)
                C(i,j) += v[i] * tmp;
        }
    }
    else {
        auto w = slice(work,pair{0,m});
        copy( col(C, 0), w );
        if(n > 1){
            auto x = slice(v,pair{1,n});
            gemv(Op::NoTrans, one, cols(C, pair{1,n}), x, one, w);
        }
        for (idx_t j = 0; j < n; ++j) {
            T tmp;
            if( j == 0 )
                tmp = -tau;
            else
                tmp = -tau * conj( v[j] );
            for (idx_t i = 0; i < m; ++i)
                C(i,j) += w[i] * tmp;
        }
    }
}

} // lapack

#endif // __LARF_HH__
