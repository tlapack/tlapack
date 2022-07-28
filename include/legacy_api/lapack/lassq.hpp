/// @file lassq.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// 
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LASSQ_HH
#define TLAPACK_LEGACY_LASSQ_HH

#include "lapack/lassq.hpp"

namespace tlapack {

/** Updates a sum of squares represented in scaled form.
 * \[
 *      scl_{[OUT]}^2 sumsq_{[OUT]} = \sum_{i = 0}^n x_i^2 + scl_{[IN]}^2 sumsq_{[IN]},
 * \]
 * The value of  sumsq  is assumed to be non-negative.
 * 
 * If (scale * sqrt( sumsq )) > tbig on entry then
 *    we require:   scale >= sqrt( TINY*EPS ) / sbig   on entry,
 * and if 0 < (scale * sqrt( sumsq )) < tsml on entry then
 *    we require:   scale <= sqrt( HUGE ) / ssml       on entry,
 * where
 *    tbig -- upper threshold for values whose square is representable;
 *    sbig -- scaling constant for big numbers; @see base/constants.hpp
 *    tsml -- lower threshold for values whose square is representable;
 *    ssml -- scaling constant for small numbers; @see base/constants.hpp
 * and
 *    TINY*EPS -- tiniest representable number;
 *    HUGE     -- biggest representable number.
 * 
 * @param[in] n The number of elements to be used from the vector x.
 * @param[in] x Array of dimension $(1+(n-1)*|incx|)$.
 * @param[in] incx. The increment between successive values of the vector x.
 *          If incx > 0, X(i*incx) = x_i for 0 <= i < n
 *          If incx < 0, X((n-i-1)*(-incx)) = x_i for 0 <= i < n
 *          If incx = 0, x isn't a vector so there is no need to call
 *          this subroutine.  If you call it anyway, it will count x_0
 *          in the vector norm n times.
 * @param[in] scl
 * @param[in] sumsq
 * 
 * @ingroup auxiliary
 */
template< typename TX >
void lassq(
    idx_t n,
    TX const* x, int_t incx,
    real_type<TX> &scl,
    real_type<TX> &sumsq)
{
    // quick return
    if( isnan(scl) || isnan(sumsq) || n <= 0 ) return;

    tlapack_expr_with_vector( x_, TX, n, x, incx, return lassq( x_, scl, sumsq ) );
}

} // lapack

#endif // TLAPACK_LEGACY_LASSQ_HH
