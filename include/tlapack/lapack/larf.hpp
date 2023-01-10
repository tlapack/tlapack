/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARF_HH
#define TLAPACK_LARF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/ger.hpp"

namespace tlapack {

/** Worspace query.
 * 
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 */
template< class side_t, class direction_t, class vector_t, class tau_t, class matrix_t >
inline constexpr
void larf_worksize(
    side_t side, direction_t direction,
    vector_t const& v, const tau_t& tau, matrix_t& C,
    workinfo_t& workinfo, const workspace_opts_t<>& opts = {} )
{
    using work_t    = vector_type< matrix_t, vector_t >;
    using idx_t     = size_type< matrix_t >;
    using T         = type_t< work_t >;

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    const workinfo_t myWorkinfo( sizeof(T), (side == Side::Left) ? n : m );
    workinfo.minMax( myWorkinfo );
}

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - \tau v v^H.
 * \]
 * where v = [ 1 x ] if direction == Direction::Forward and
 *       v = [ x 1 ] if direction == Direction::Backward.
 * 
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * 
 * @param[in] side
 *     - Side::Left:  apply $H$ from the Left.
 *     - Side::Right: apply $H$ from the Right.
 *
 * @param[in] direction
 *     v = [ 1 x ] if direction == Direction::Forward and
 *     v = [ x 1 ] if direction == Direction::Backward.
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
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 * 
 * @ingroup auxiliary
 */
template< class side_t, class direction_t, class vector_t, class tau_t, class matrix_t >
void larf(
    side_t side, direction_t direction,
    vector_t const& v, const tau_t& tau,
    matrix_t& C, const workspace_opts_t<>& opts = {} )
{
    // data traits
    using work_t    = vector_type< matrix_t, vector_t >;
    using idx_t     = size_type< matrix_t >;
    using T         = type_t< work_t >;
    using real_t    = real_type<T>;

    using pair = pair<idx_t,idx_t>;

    // Functor
    Create<work_t> new_vector;

    // constants
    const real_t one(1);
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);

    // check arguments
    tlapack_check( side == Side::Left ||
                   side == Side::Right );
    tlapack_check( direction == Direction::Backward ||
                   direction == Direction::Forward );
    tlapack_check(  access_granted( dense, write_policy(C) ) );

    // Allocates workspace
    vectorOfBytes localworkdata;
    const Workspace work = [&]()
    {
        workinfo_t workinfo;
        larf_worksize( side, direction, v, tau, C, workinfo, opts );
        return alloc_workspace( localworkdata, workinfo, opts.work );
    }();

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

    if( side == Side::Left )
    {
        if( m <= 1 )
        {
            for (idx_t j = 0; j < n; ++j)
                C(0,j) -= tau*C(0,j);
            return;
        }

        auto C0 = (direction == Direction::Forward) ? row(C,0) : row(C,m-1);
        auto C1 = (direction == Direction::Forward) ? rows(C, pair{1,m}) : rows(C, pair{0,m-1});
        auto x  = (direction == Direction::Forward) ? slice(v, pair{1,m}) : slice(v, pair{0,m-1});
        auto w  = new_vector( work, n );

        // w := C0^H + C1^H*x
        for (idx_t i = 0; i < n; ++i )
            w[i] = conj(C0[i]);
        gemv(Op::ConjTrans, one, C1, x, one, w);

        // C1 := C1 - tau*x*w^H
        ger( -tau, x, w, C1 );

        // C0 := C0 - tau*w^H
        for (idx_t i = 0; i < n; ++i )
            C0[i] -= tau*conj(w[i]);
    }
    else
    {
        if( n <= 1 )
        {
            for (idx_t i = 0; i < m; ++i)
                C(i,0) -= tau*C(i,0);
            return;
        }

        auto C0 = (direction == Direction::Forward) ? col(C,0) : col(C,n-1);
        auto C1 = (direction == Direction::Forward) ? cols(C, pair{1,n}) : cols(C, pair{0,n-1});
        auto x  = (direction == Direction::Forward) ? slice(v, pair{1,n}) : slice(v, pair{0,n-1});
        auto w = new_vector( work, m );

        // w := C0 + C1*x
        for (idx_t i = 0; i < m; ++i )
            w[i] = C0[i];
        gemv(Op::NoTrans, one, C1, x, one, w);

        // C1 := C1 - tau*w*x^H
        ger( -tau, w, x, C1 );

        // C0 := C0 - tau*w
        for (idx_t i = 0; i < m; ++i )
            C0[i] -= tau*w[i];
    }
}

} // lapack

#endif // TLAPACK_LARF_HH
