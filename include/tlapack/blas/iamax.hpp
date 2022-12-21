// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_IAMAX_HH
#define TLAPACK_BLAS_IAMAX_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/base/constants.hpp"

namespace tlapack {

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 * 
 * Version with NaN checks.
 * @see iamax_nc( const vector_t& x ) for the version that does not check for NaNs.
 * 
 * @param[in] x The n-element vector x.
 * 
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `NAN` in $x$ if it exists,
 * 3. the index of the first `Infinity` in $x$ if it exists,
 * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
 *
 * @ingroup iamax
 */
template< class vector_t >
size_type<vector_t>
iamax_ec( const vector_t& x )
{
    // data traits
    using idx_t  = size_type< vector_t >;
    using T      = type_t<vector_t>;
    using real_t = real_type< T >;

    // constants
    const real_t oneFourth( 0.25f );
    const idx_t n = size(x);

    // quick return
    if( n <= 0 ) return 0;

    bool scaledsmax = false; // indicates whether |Re(x_i)| + |Im(x_i)| = Inf
    real_t smax( -1.0f );
    idx_t index = -1;
    idx_t i = 0;
    for (; i < n; ++i) {
        if ( isnan(x[i]) ) {
            // return when first NaN found
            return i;
        }
        else if ( isinf(x[i]) ) {
            
            // keep looking for first NaN
            for (idx_t k = i+1; k < n; ++k) {
                if ( isnan(x[k]) ) {
                    // return when first NaN found
                    return k;
                }
            }

            // return the position of the first Inf
            return i;
        }
        else { // still no Inf found yet
            if ( ! is_complex<T>::value ) {
                real_t a = abs1(x[i]);
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            else if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
                real_t a = abs1(x[i]);
                if ( isinf(a) ) {
                    scaledsmax = true;
                    smax = abs1( oneFourth*x[i] );
                    index = i;
                }
                else if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            else { // scaledsmax = true
                real_t a = abs1( oneFourth*x[i] );
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
        }
    }

    return index;
}

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 * 
 * Version with no NaN checks.
 * @see iamax_ec( const vector_t& x ) for the version that check for NaNs.
 * 
 * @param[in] x The n-element vector x.
 * 
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `Infinity` in $x$ if it exists,
 * 3. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
 *
 * @ingroup iamax
 */
template< class vector_t >
size_type<vector_t>
iamax_nc( const vector_t& x )
{
    // data traits
    using idx_t  = size_type< vector_t >;
    using T      = type_t<vector_t>;
    using real_t = real_type< T >;

    // constants
    const real_t oneFourth( 0.25f );
    const idx_t n = size(x);

    // quick return
    if( n <= 0 ) return 0;

    bool scaledsmax = false; // indicates whether |Re(x_i)| + |Im(x_i)| = Inf
    real_t smax( -1.0f );
    idx_t index = -1;
    idx_t i = 0;
    for (; i < n; ++i) {
        if ( isinf(x[i]) ) {
            // return the position of the first Inf
            return i;
        }
        else { // still no Inf found yet
            if ( ! is_complex<T>::value ) {
                real_t a = abs1(x[i]);
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            else if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
                real_t a = abs1(x[i]);
                if ( isinf(a) ) {
                    scaledsmax = true;
                    smax = abs1( oneFourth*x[i] );
                    index = i;
                }
                else if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            else { // scaledsmax = true
                real_t a = abs1( oneFourth*x[i] );
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
        }
    }

    return (index != idx_t(-1)) ? index : 0;
}

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 * 
 * @see iamax_nc( const vector_t& x ) for the version that does not check for NaNs.
 * @see iamax_ec( const vector_t& x ) for the version that check for NaNs.
 * 
 * @param[in] x The n-element vector x.
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs.
 * 
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `NAN` in $x$ if it exists and if `ec.nan == true`,
 * 3. the index of the first `Infinity` in $x$ if it exists,
 * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
 *
 * @ingroup iamax
 */
template< class vector_t, 
    disable_if_allow_optblas_t< vector_t > = 0
>
inline
size_type<vector_t>
iamax( const vector_t& x, const ec_opts_t& opts = {} )
{
    return ( opts.ec.nan == true ) ? iamax_ec(x) : iamax_nc(x);
}

#ifdef USE_LAPACKPP_WRAPPERS

    template< class vector_t,
        enable_if_allow_optblas_t< vector_t > = 0
    >
    inline
    auto iamax( vector_t const& x )
    {
        auto x_ = legacy_vector(x);
        return ::blas::iamax( x_.n, x_.ptr, x_.inc );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_IAMAX_HH
