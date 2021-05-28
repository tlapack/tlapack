// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
// adapting https://github.com/langou/latl/blob/master/include/larf.h from Rodney James, University of Colorado Denver, USA.

#ifndef __LARF_HH__
#define __LARF_HH__

#include "lapack/types.hpp"

#include "blas/gemv.hpp"
#include "blas/ger.hpp"

namespace lapack {

/** Applies an elementary reflector H to a m-by-n matrix C.
 *
 * The elementary reflector H can be applied on either the left or right, with
 * \[
 *        H = I - tau v v^H.
 * \]
 * If tau = 0, then H is taken to be the unit matrix.
 * 
 * @param side Specifies whether the elementary reflector H is applied on the left or right.

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
 * @param w Workspace vector of of the following length:
 *
 *          n if side='L'
 *          m if side='R'.
 * 
 * @ingroup auxiliary
 */
template< typename TV, typename TC, typename TW >
inline void larf(
    blas::Side side,
    blas::size_t m, blas::size_t n,
    TV const *v, blas::int_t incv,
    blas::scalar_type< TV, TC , TW > tau,
    TC *C, blas::int_t ldC,
    TW *w )
{
    typedef blas::real_type<TV, TC, TW> real_t;

    // constants
    const real_t one(1.0);
    const real_t zero(0.0);

    if ( side == Side::Left ) {
        gemv(Layout::ColMajor, Op::ConjTrans, m, n, one, C, ldC, v, incv, zero, w, 1);
        ger(m, n, -tau, v, incv, w, 1, C, ldC);
    }
    else if ( side == Side::Right ) {
        gemv(Layout::ColMajor, Op::NoTrans, m, n, one, C, ldC, v, incv, zero, w, 1);
        ger(m, n, -tau, w, 1, v, incv, C, ldC);
    }
    else {
        blas_error( "side != Side::Left && side != Side::Right" );
    }
}

}

#endif // __LARF_HH__