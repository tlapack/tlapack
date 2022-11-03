/// @file nrm2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
///
/// Anderson E. (2017)
/// Algorithm 978: Safe Scaling in the Level 1 BLAS
/// ACM Trans Math Softw 44:1--28
/// @see https://doi.org/10.1145/3061665
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_NRM2_HH
#define TLAPACK_BLAS_NRM2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/base/constants.hpp"

namespace tlapack {

/**
 * @return 2-norm of vector,
 *     $|| x ||_2 := (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
 *
 * @param[in] x A n-element vector.
 *
 * @ingroup nrm2
 */
template< class vector_t,
    disable_if_allow_optblas_t< vector_t > = 0
>
auto nrm2( const vector_t& x )
{
    using real_t = real_type< type_t<vector_t> >;
    using idx_t  = size_type< vector_t >;

    // constants
    const idx_t n = size(x);

    // constants
    const real_t zero( 0 );
    const real_t one( 1 );
    const real_t tsml = blue_min<real_t>();
    const real_t tbig = blue_max<real_t>();
    const real_t ssml = blue_scalingMin<real_t>();
    const real_t sbig = blue_scalingMax<real_t>();

    // scaled sum of squares
    real_t scl = one;
    real_t sumsq = zero;

    // quick return
    if( n <= 0 ) return zero;

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

    for (idx_t i = 0; i < n; ++i)
    {
        real_t ax = tlapack::abs( x[i] );
        if( ax > tbig )
            abig += (ax*sbig) * (ax*sbig);
        else if( ax < tsml ) {
            if( abig == zero ) asml += (ax*ssml) * (ax*ssml);
        } else
            amed += ax * ax;
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
            sumsq = (ymax * ymax) * ( one + (ymin/ymax) * (ymin/ymax) );
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

    return scl * sqrt( sumsq );
}

#ifdef USE_LAPACKPP_WRAPPERS

    template< class vector_t,
        enable_if_allow_optblas_t< vector_t > = 0
    >
    inline
    auto nrm2( vector_t const& x )
    {
        auto x_ = legacy_vector(x);
        return ::blas::nrm2( x_.n, x_.ptr, x_.inc );
    }

#endif

}  // namespace tlapack

#endif        // #ifndef TLAPACK_BLAS_NRM2_HH
