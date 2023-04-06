/// @file lassq.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LASSQ_HH
#define TLAPACK_LASSQ_HH

#include "tlapack/base/constants.hpp"
#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Updates a sum of squares represented in scaled form.
 * \[
 *      scl smsq := \sum_{i = 0}^n |x_i|^2 + scale^2 sumsq,
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
 * @param[in] x Vector of size n.
 *
 * @param[in,out] scale Real scalar.
 *      On entry, the value `scale` in the equation above.
 *      On exit, `scale` is overwritten with `scl`, the scaling factor
 *      for the sum of squares.
 *
 * @param[in,out] sumsq Real scalar.
 *      On entry, the value `sumsq` in the equation above.
 *      On exit, `sumsq` is overwritten with `smsq`, the basic sum of
 *      squares from which  scl  has been factored out.
 *
 * @param[in] absF Lambda function that computes the absolute value.
 * \code{.cpp}
 *      absF = []( const T& x ) -> real_type<T> { ... };
 * \endcode
 *
 * @ingroup auxiliary
 */
template <class abs_f, TLAPACK_VECTOR vector_t, class T = type_t<vector_t> >
void lassq(const vector_t& x,
           real_type<T>& scale,
           real_type<T>& sumsq,
           abs_f absF)
{
    using real_t = real_type<T>;
    using idx_t = size_type<vector_t>;

    // constants
    const idx_t n = size(x);

    // constants
    const real_t zero(0);
    const real_t one(1);
    const real_t tsml = blue_min<real_t>();
    const real_t tbig = blue_max<real_t>();
    const real_t ssml = blue_scalingMin<real_t>();
    const real_t sbig = blue_scalingMax<real_t>();

    // quick return
    if (isnan(scale) || isnan(sumsq)) return;

    if (sumsq == zero) scale = one;
    if (scale == zero) {
        scale = one;
        sumsq = zero;
    }

    // quick return
    if (n <= 0) return;

    //  Compute the sum of squares in 3 accumulators:
    //     abig -- sums of squares scaled down to avoid overflow
    //     asml -- sums of squares scaled up to avoid underflow
    //     amed -- sums of squares that do not require scaling
    //  The thresholds and multipliers are
    //     tbig -- values bigger than this are scaled down by sbig
    //     tsml -- values smaller than this are scaled up by ssml

    real_t asml = zero;
    real_t amed = zero;
    real_t abig = zero;

    for (idx_t i = 0; i < n; ++i) {
        real_t ax = absF(x[i]);
        if (ax > tbig)
            abig += (ax * sbig) * (ax * sbig);
        else if (ax < tsml) {
            if (abig == zero) asml += (ax * ssml) * (ax * ssml);
        }
        else
            amed += ax * ax;
    }

    // Put the existing sum of squares into one of the accumulators
    if (sumsq > zero) {
        real_t ax = scale * sqrt(sumsq);
        if (ax > tbig)
            abig += ((scale * sbig) * (scale * sbig)) * sumsq;
        else if (ax < tsml) {
            if (abig == zero) asml += ((scale * ssml) * (scale * ssml)) * sumsq;
        }
        else
            amed += (scale * scale) * sumsq;
    }

    // Combine abig and amed or amed and asml if
    // more than one accumulator was used.

    if (abig > zero) {
        // Combine abig and amed if abig > 0
        if (amed > zero || isnan(amed)) abig += (amed * sbig) * sbig;
        scale = one / sbig;
        sumsq = abig;
    }
    else if (asml > zero) {
        // Combine amed and asml if asml > 0
        if (amed > zero || isnan(amed)) {
            amed = sqrt(amed);
            asml = sqrt(asml) / ssml;

            real_t ymin, ymax;
            if (asml > amed) {
                ymin = amed;
                ymax = asml;
            }
            else {
                ymin = asml;
                ymax = amed;
            }

            scale = one;
            sumsq = (ymax * ymax) * (one + (ymin / ymax) * (ymin / ymax));
        }
        else {
            scale = one / ssml;
            sumsq = asml;
        }
    }
    else {
        // Otherwise all values are mid-range or zero
        scale = one;
        sumsq = amed;
    }
}

/** Updates a sum of squares represented in scaled form.
 * \[
 *      scl smsq := \sum_{i = 0}^n |x_i|^2 + scale^2 sumsq.
 * \]
 * @see lassq(
    const vector_t& x,
    real_type<T> &scale,
    real_type<T> &sumsq,
    abs_f absF ).
 *
 * Specific implementation using
 * \code{.cpp}
 *      absF = []( const T& x ) { return tlapack::abs( x ); }
 * \endcode
 * where T is the type_t< vector_t >.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_VECTOR vector_t, class T = type_t<vector_t> >
inline void lassq(const vector_t& x, real_type<T>& scale, real_type<T>& sumsq)
{
    return lassq(
        x, scale, sumsq,
        // Lambda function that returns the absolute value using tlapack::abs :
        [](const T& x) { return tlapack::abs(x); });
}

}  // namespace tlapack

#endif  // TLAPACK_LASSQ_HH
