// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "types.hpp"
#include "exception.hpp"
#include "utils.hpp"
#include "constants.hpp"

#include <limits>

namespace blas {

// =============================================================================
/// @return Index of infinity-norm of vector, $|| x ||_{inf}$,
///     $\text{argmax}_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
/// Returns -1 if n = 0.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup iamax

// Real case
template< typename T >
size_t iamax(
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
                break;
            }
            else { // still no Inf found yet
                real_t a = std::abs(x[ix]);
                if ( a > smax ) { // and everything finite so far
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

// Complex case
template< typename T >
size_t iamax(
    size_t n,
    std::complex<T> const *x, int_t incx )
{
    typedef T real_t;

    bool scaledsmax = false; // indicates whether sx(i) finite but abs(real(sx(i))) + abs(imag(sx(i))) = Inf
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
                break;
            }
            else { // still no Inf found yet
                if ( !scaledsmax ) { // no abs(real(sx(i))) + abs(imag(sx(i))) = Inf  yet
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
                break;
            }
            else { // still no Inf found yet
                if ( !scaledsmax ) { // no abs(real(sx(i))) + abs(imag(sx(i))) = Inf  yet
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
