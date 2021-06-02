/// @file larf.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larf.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of T-LAPACK.
// T-LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARF_HH__
#define __LARF_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

#include "blas/gemv.hpp"
#include "blas/ger.hpp"

namespace lapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - \tau v v^H.
 * \]
 * If tau = 0, then H is taken to be the unit matrix.
 * 
 * @returns  0 if success.
 * @returns -1 if side is unknown.
 * 
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 * 
 * @param[in] side Specifies whether the elementary reflector H is applied on the left or right.
 *
 *              side='L': form  H * C
 *              side='R': form  C * H
 * 
 * @param[in] m Number of rows of the matrix C.
 * @param[in] n Number of columns of the matrix C.
 * @param[in] v Vector of containing the elementary reflector.
 *
 *              If side='R', v is of length n.
 *              If side='L', v is of length m.
 * 
 * @param[in] incv Increment of the vector v.
 * @param[in] tau Value of tau in the representation of H.
 * @param[in,out] C m-by-n matrix.  On exit, C is overwritten with
 *
 *                H * C if side='L',
 *             or C * H if side='R'.
 * 
 * @param[in] ldC Column length of matrix C.  ldC >= m.
 * @param work Workspace vector of of the following length:
 *
 *          n if side='L'
 *          m if side='R'.
 * 
 * @ingroup auxiliary
 */
template< typename TV, typename TC, typename TW >
inline int larf(
    Layout layout, Side side,
    blas::size_t m, blas::size_t n,
    TV const *v, blas::int_t incv,
    blas::scalar_type< TV, TC , TW > tau,
    TC *C, blas::size_t ldC,
    TW *work )
{
    typedef blas::real_type<TV, TC, TW> real_t;
    using blas::gemm;
    using blas::ger;

    // constants
    const real_t one(1.0);
    const real_t zero(0.0);

    if ( side == Side::Left ) {
        gemv(layout, Op::ConjTrans, m, n, one, C, ldC, v, incv, zero, work, 1);
        ger(layout, m, n, -tau, v, incv, work, 1, C, ldC);
    }
    else if ( side == Side::Right ) {
        gemv(layout, Op::NoTrans, m, n, one, C, ldC, v, incv, zero, work, 1);
        ger(layout, m, n, -tau, work, 1, v, incv, C, ldC);
    }
    else
        lapack_error( "side != Side::Left && side != Side::Right", -1 );

    return 0;
}

/** Applies an elementary reflector H to a m-by-n matrix C.
 * 
 * @see larf( Layout, Side side, blas::size_t m, blas::size_t n, TV const *v, blas::int_t incv, blas::scalar_type< TV, TC , TW > tau, TC *C, blas::size_t ldC, TW *work )
 * 
 * @ingroup auxiliary
 */
template< typename TV, typename TC, typename TW >
inline int larf(
    Side side,
    blas::size_t m, blas::size_t n,
    TV const *v, blas::int_t incv,
    blas::scalar_type< TV, TC , TW > tau,
    TC *C, blas::size_t ldC,
    TW *work )
{
    return larf(
        Layout::ColMajor, side, m, n, v, incv, tau, C, ldC, work );
}

} // lapack

#endif // __LARF_HH__