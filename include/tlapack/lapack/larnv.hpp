/// @file larnv.hpp Returns a vector of random numbers from a uniform or normal distribution.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larnv.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARNV_HH
#define TLAPACK_LARNV_HH

#include <random>
#include "tlapack/base/types.hpp"

namespace tlapack {

/** Returns a vector of n random numbers from a uniform or normal distribution.
 * 
 * This implementation uses the Mersenne Twister 19937 generator (std::mt19937),
 * which is a Mersenne Twister pseudo-random generator of 32-bit numbers with
 * a state size of 19937 bits.
 * 
 * Requires ISO C++ 2011 random number generators.
 * 
 * @tparam idist Specifies the distribution:
 *      1.  real and imaginary parts each uniform (0,1).
 *      2.  real and imaginary parts each uniform (-1,1).
 *      3.  real and imaginary parts each normal (0,1).
 *      4.  uniformly distributed on the disc abs(z) < 1.
 *      5.  uniformly distributed on the circle abs(z) = 1.
 * 
 * @param[in,out] iseed Seed for the random number generator.
 *      The seed is updated inside the routine ( seed := seed + 1 ).
 * 
 * @param[out] x Vector of length n.
 * 
 * @ingroup auxiliary
 */
template< int idist, class vector_t, class Sseq >
void larnv( Sseq& iseed, vector_t& x )
{
    using idx_t  = size_type< vector_t >;
    using T      = type_t< vector_t >;
    using real_t = real_type< T >;


    // Constants
    const idx_t n = size(x);
    const real_t one(1);
    const real_t eight(8);
    const real_t twopi = eight * atan(one);

    // Initialize the Mersenne Twister generator
    std::random_device device;
    std::mt19937 generator(device());
    generator.seed(iseed);

    if (idist == 1) {
        std::uniform_real_distribution<real_t> d1(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( is_complex<T>::value )
                x[i] = make_scalar<T>( d1(generator), d1(generator) );
            else
                x[i] = d1(generator);
        }
    }
    else if (idist == 2) {
        std::uniform_real_distribution<real_t> d2(-1, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( is_complex<T>::value )
                x[i] = make_scalar<T>( d2(generator), d2(generator) );
            else
                x[i] = d2(generator);
        }
    }
    else if (idist == 3) {
        std::normal_distribution<real_t> d3(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( is_complex<T>::value )
                x[i] = make_scalar<T>( d3(generator), d3(generator) );
            else
                x[i] = d3(generator);
        }
    }
    else if ( is_complex<T>::value ) {
        if (idist == 4) {
            std::uniform_real_distribution<real_t> d4(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                real_t r     = sqrt(d4(generator));
                real_t theta = twopi * d4(generator);
                x[i] = make_scalar<T>( r*cos(theta), r*sin(theta) );
            }
        }
        else if (idist == 5) {
            std::uniform_real_distribution<real_t> d5(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                real_t theta = twopi * d5(generator);
                x[i] = make_scalar<T>( cos(theta), sin(theta) );
            }
        }
    }

    // Update the seed
    iseed = iseed + 1;
}

}

#endif // TLAPACK_LARNV_HH
