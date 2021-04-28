// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "blas/types.hpp"
#include "exception.hpp"
#include "blas/utils.hpp"
#include "blas/constants.hpp"

namespace blas {

// =============================================================================
/// @return In priority order:
/// 1. INVALID_INDEX if $n \le 0$,
/// 2. the index of the first `NAN` in $x$ if it exists and if `checkInput == true`,
/// 3. the index of the first `Infinity` in $x$ if it exists,
/// 4. the Index of infinity-norm of $x$, $|| x ||_{inf}$,
///     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @param[in] checkInput
///     If true, check for invalid values in the input.
///
/// @ingroup iamax

template< typename T >
size_t iamax(
    size_t n,
    T const *x, int_t incx,
    bool checkInput = true );

/// @ingroup iamax
/// GIGO: Garbage In Garbage Out
///
template< typename T >
size_t iamax_GIGO(
    size_t n,
    T const *x, int_t incx );

/// @ingroup iamax
/// SAFE: Check for Infs and NaNs
///
template< typename T >
size_t iamax_SAFE(
    size_t n,
    T const *x, int_t incx );

// =============================================================================
// Implementation

template< typename T >
size_t iamax(
    size_t n,
    T const *x, int_t incx,
    bool checkInput ) {

    if ( checkInput ) {
        blas_error_if( incx <= 0 );
        return iamax_SAFE( n, x, incx );
    } else
        return iamax_GIGO( n, x, incx );
}

template< typename T >
size_t iamax_GIGO(
    size_t n,
    T const *x, int_t incx )
{    
    typedef real_type<T> real_t;

    size_t index = INVALID_INDEX;
    real_t smax = -1;

    if (incx == 1) {
        // unit stride
        for (size_t i = 0; i < n; ++i) {
            real_t a = std::abs(x[i]);
            if ( a > smax ) {
                smax = a;
                index = i;
            }
        }
    }
    else {
        // non-unit stride
        int_t ix = 0;
        for (size_t i = 0; i < n; ++i) {
            real_t a = std::abs(x[ix]);
            if ( a > smax ) {
                smax = a;
                index = i;
            }
            ix += incx;
        }
    }
    return index;
}

template< typename T >
size_t iamax_SAFE(
    size_t n,
    T const *x, int_t incx )
{    
    typedef real_type<T> real_t;

    size_t index = INVALID_INDEX;
    real_t smax = -1;

    if (incx == 1) {
        // unit stride
        size_t i = 0;
        for (; i < n; ++i) {
            if ( isnan(x[i]) ) {
                // return when first NaN found
                return i;
            }
            else if ( isinf(x[i]) ) {
                // record location of first Inf
                index = i;
                i++;
                break;
            }
            else { // still no Inf found yet
                real_t a = std::abs(x[i]);
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
        }
        for (; i < n; ++i) { // keep looking for first NaN
            if ( isnan(x[i]) ) {
                // return when first NaN found
                return i;
            }
        }
    }
    else {
        // non-unit stride
        size_t i = 0;
        int_t ix = 0;
        for (; i < n; ++i) {
            if ( isnan(x[ix]) ) {
                // return when first NaN found
                return i;
            }
            else if ( isinf(x[ix]) ) {
                // record location of first Inf
                index = i;
                i++; ix += incx;
                break;
            }
            else { // still no Inf found yet
                real_t a = std::abs(x[ix]);
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            ix += incx;
        }
        for (; i < n; ++i) { // keep looking for first NaN
            if ( isnan(x[ix]) ) {
                // return when first NaN found
                return i;
            }
            ix += incx;
        }
    }
    return index;
}

template< typename T >
size_t iamax_GIGO(
    size_t n,
    std::complex<T> const *x, int_t incx )
{
    typedef T real_t;

    bool scaledsmax = false; // indicates whether |Re(x_i)| + |Im(x_i)| = Inf
    real_t smax = -1;
    size_t index = INVALID_INDEX;
    const real_t oneFourth = 0.25;

    if (incx == 1) {
        // unit stride
        for (size_t i = 0; i < n; ++i) {
            if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
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
    else {
        // non-unit stride
        int_t ix = 0;
        for (size_t i = 0; i < n; ++i) {
            if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
                real_t a = abs1(x[ix]);
                if ( isinf(a) ) {
                    scaledsmax = true;
                    smax = abs1( oneFourth*x[ix] );
                    index = i;
                }
                else if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            else { // scaledsmax = true
                real_t a = abs1( oneFourth*x[ix] );
                if ( a > smax ) {
                    smax = a;
                    index = i;
                }
            }
            ix += incx;
        }
    }
    return index;
}

template< typename T >
size_t iamax_SAFE(
    size_t n,
    std::complex<T> const *x, int_t incx )
{
    typedef T real_t;

    bool scaledsmax = false; // indicates whether x_i finite but |Re(x_i)| + |Im(x_i)| = Inf
    real_t smax = -1;
    size_t index = INVALID_INDEX;
    const real_t oneFourth = 0.25;

    if (incx == 1) {
        // unit stride
        size_t i = 0;
        for (; i < n; ++i) {
            if ( isnan(x[i]) ) {
                // return when first NaN found
                return i;
            }
            else if ( isinf(x[i]) ) {
                // record location of first Inf
                index = i;
                i++;
                break;
            }
            else { // still no Inf found yet
                if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
                    real_t a = abs1(x[i]);
                    if ( isinf(a) ) {
                        scaledsmax = true;
                        smax = abs1( oneFourth*x[i] );
                        index = i;
                    }
                    else if ( a > smax ) { // and everything finite so far
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
        for (; i < n; ++i) { // keep looking for first NaN
            if ( isnan(x[i]) ) {
                // return when first NaN found
                return i;
            }
        }
    }
    else {
        // non-unit stride
        size_t i = 0;
        int_t ix = 0;
        for (; i < n; ++i) {
            if ( isnan(x[ix]) ) {
                // return when first NaN found
                return i;
            }
            else if ( isinf(x[ix]) ) {
                // record location of first Inf
                index = i;
                i++; ix += incx;
                break;
            }
            else { // still no Inf found yet
                if ( !scaledsmax ) { // no |Re(x_i)| + |Im(x_i)| = Inf  yet
                    real_t a = abs1(x[ix]);
                    if ( isinf(a) ) {
                        scaledsmax = true;
                        smax = abs1( oneFourth*x[ix] );
                        index = i;
                    }
                    else if ( a > smax ) { // and everything finite so far
                        smax = a;
                        index = i;
                    }
                }
                else { // scaledsmax = true
                    real_t a = abs1( oneFourth*x[ix] );
                    if ( a > smax ) {
                        smax = a;
                        index = i;
                    }
                }
            }
            ix += incx;
        }
        for (; i < n; ++i) { // keep looking for first NaN
            if ( isnan(x[ix]) ) {
                // return when first NaN found
                return i;
            }
            ix += incx;
        }
    }
    return index;
}

}  // namespace blas

#endif        //  #ifndef BLAS_IAMAX_HH
