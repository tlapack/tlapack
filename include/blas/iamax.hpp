// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "blas/utils.hpp"
#include "blas/constants.hpp"

namespace blas {

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 *
 * @param[in] c Tells the algorithm how to check the input.
 *      Either nocheck or checkInfNaN.
 * 
 * @param[in] x The n-element vector x.
 * 
 * @return In priority order:
 * 1. 0 if n <= 0,
 * 2. the index of the first `NAN` in $x$ if it exists and if `checkNAN == true`,
 * 3. the index of the first `Infinity` in $x$ if it exists,
 * 4. the Index of the infinity-norm of $x$, $|| x ||_{inf}$,
 *     $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$.
 *
 * @ingroup iamax
 */
template< class vector_t, class check_t,
    enable_if_t<(
    /* Requires: */
        is_same_v< check_t, nocheck_t > || 
        is_same_v< check_t, checkInfNaN_t >
    ), int > = 0 >
size_type<vector_t>
iamax( check_t c, const vector_t& x )
{
    // data traits
    using idx_t  = size_type< vector_t >;
    using T      = type_t<vector_t>;
    using real_t = real_type< T >;

    // constants
    const real_t oneFourth = 0.25;
    const idx_t n = size(x);
    constexpr bool check = is_same_v< check_t, checkInfNaN_t >;

    bool scaledsmax = false; // indicates whether |Re(x_i)| + |Im(x_i)| = Inf
    real_t smax = -1;
    idx_t index = -1;
    idx_t i = 0;
    for (; i < n; ++i) {
        if ( check && isnan(x[i]) ) {
            // return when first NaN found
            return i;
        }
        else if ( check && isinf(x[i]) ) {
            // record location of first Inf
            index = i;
            i++;
            break;
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
    if ( check ) {
        for (; i < n; ++i) { // keep looking for first NaN
            if ( isnan(x[i]) ) {
                // return when first NaN found
                return i;
            }
        }
    }

    return (index != idx_t(-1)) ? index : 0;
}

/**
 * @brief Return $\arg\max_{i=0}^{n-1} \left(|Re(x_i)| + |Im(x_i)|\right)$
 * 
 * @see iamax( check_t c, const vector_t& x )
 * 
 * @ingroup iamax
 */
template< class vector_t >
inline auto
iamax( const vector_t& x ) { return iamax( checkInfNaN, x ); }

}  // namespace blas

#endif        //  #ifndef BLAS_IAMAX_HH
