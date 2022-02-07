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

#ifndef __SLATE_LASSQ_HH__
#define __SLATE_LASSQ_HH__

#include "lapack/lassq.hpp"

namespace lapack {

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
 *    sbig -- scaling constant for big numbers; @see blas/constants.hpp
 *    tsml -- lower threshold for values whose square is representable;
 *    ssml -- scaling constant for small numbers; @see blas/constants.hpp
 * and
 *    TINY*EPS -- tiniest representable number;
 *    HUGE     -- biggest representable number.
 * 
 * @param[in] n The number of elements to be used from the vector x.
 * @param[in] x Array of dimension $(1+(n-1)*\abs(incx))$.
 * @param[in] incx. The increment between successive values of the vector x.
 *          If incx > 0, X(i*incx) = x_i for 0 <= i < n
 *          If incx < 0, X((n-i-1)*(-incx)) = x_i for 0 <= i < n
 *          If incx = 0, x isn't a vector so there is no need to call
 *          this subroutine.  If you call it anyway, it will count x_0
 *          in the vector norm n times.
 * @param[in] scl
 * @param[in] sumsq
 * 
 * @ingroup
 */
template< typename TX >
void lassq(
    blas::idx_t n,
    TX const* x, blas::int_t incx,
    real_type<TX> &scl,
    real_type<TX> &sumsq)
{
    using real_t = real_type<TX>;
    using blas::internal::vector;
    using blas::isnan;

    // constants
    const real_t zero( 0 );
    const real_t one( 1 );

    // quick return
    if( isnan(scl) || isnan(sumsq) ) return;

    if( sumsq == zero ) scl = one;
    if( scl == zero ) {
        scl = one;
        sumsq = zero;
    }

    // quick return
    if( n <= 0 ) return;

    const auto _x = vector<TX>( (TX*) x, n, incx );
    lassq( _x, scl, sumsq );
}

} // lapack

#endif // __LASSQ_HH__
