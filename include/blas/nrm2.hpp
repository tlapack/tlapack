/// @file nrm2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_NRM2_HH
#define BLAS_NRM2_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * @return 2-norm of vector,
 *     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup nrm2
 */
template< typename T >
real_type<T>
nrm2(
    blas::size_t n,
    T const * x, blas::int_t incx )
{
    typedef real_type<T> real_t;
    #define SQUARE(x) (x)*(x)

    // check arguments
    blas_error_if( incx <= 0 );

    // constants
    const real_t zero(0.0);
    const real_t one(1.0);
    const real_t tsml = blas::blue_min<real_t>();
    const real_t tbig = blas::blue_max<real_t>();
    const real_t ssml = blas::blue_scalingMin<real_t>();
    const real_t sbig = blas::blue_scalingMax<real_t>();

    // scaled sum of squares
    real_t scl = one;
    real_t sumsq = zero;

    // Compute the sum of squares in 3 accumulators:
    //    abig -- sums of squares scaled down to avoid overflow
    //    asml -- sums of squares scaled up to avoid underflow
    //    amed -- sums of squares that do not require scaling
    // The thresholds and multipliers are
    //    tbig -- values bigger than this are scaled down by sbig
    //    tsml -- values smaller than this are scaled up by ssml
    real_t asml = zero;
    real_t amed = zero;
    real_t abig = zero;

    if (incx == 1) {
        // unit stride
        for (size_t i = 0; i < n; ++i)
        {
            real_t ax = blas::abs( x[i] );
            if( ax > tbig )
                abig += SQUARE(ax*sbig);
            else if( ax < tsml ) {
                if( abig == zero ) asml += SQUARE(ax*asml);
            } else
                amed += SQUARE(ax);
        }
    }
    else {
        // non-unit stride
        size_t ix = 0;
        for (size_t i = 0; i < n; ++i)
        {
            real_t ax = blas::abs( x[ix] ); 
            if( ax > tbig )
                abig += SQUARE(ax*sbig);
            else if( ax < tsml ) {
                if( abig == zero ) asml += SQUARE(ax*asml);
            } else
                amed += SQUARE(ax);
            ix += incx;
        }
    }

    // Combine abig and amed or amed and asml if
    // more than one accumulator was used.

    if( abig > zero ) {
        // Combine abig and amed if abig > 0
        if( amed > zero || isnan(amed) )
            abig += (amed*sbig)*sbig;
        scl = one / sbig;
        sumsq = abig;
    }
    else if( asml > zero ) {
        // Combine amed and asml if asml > 0
        if( amed > zero || isnan(amed) ) {
            
            amed = sqrt(amed);
            asml = sqrt(asml) / ssml;
            
            real_t ymin, ymax;
            if( asml > amed ) {
                ymin = amed;
                ymax = asml;
            } else {
                ymin = asml;
                ymax = amed;
            }

            scl = one;
            sumsq = SQUARE(ymax) * ( one + SQUARE(ymin/ymax) );
        }
        else {
            scl = one / ssml;
            sumsq = asml;
        }
    }
    else {
        // Otherwise all values are mid-range or zero
        scl = one;
        sumsq = amed;
    }

    #undef SQUARE

    return scl * sqrt( sumsq );
}

}  // namespace blas

#endif        // #ifndef BLAS_NRM2_HH
