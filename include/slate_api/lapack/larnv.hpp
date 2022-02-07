/// @file larnv.hpp Returns a vector of random numbers from a uniform or normal distribution.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larnv.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_LARNV_HH__
#define __SLATE_LARNV_HH__

#include "slate_api/lapack/types.hpp"
#include "lapack/larnv.hpp"

namespace lapack {

/**
 * @brief Returns a vector of n random numbers from a uniform or normal distribution.
 * 
 * This implementation uses the Mersenne Twister 19937 generator (class std::mt19937),
 * which is a Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
 * 
 * Requires ISO C++ 2011 random number generators.
 * 
 * @param[in] idist Specifies the distribution:
 *
 *        1:  real and imaginary parts each uniform (0,1)
 *        2:  real and imaginary parts each uniform (-1,1)
 *        3:  real and imaginary parts each normal (0,1)
 *        4:  uniformly distributed on the disc abs(z) < 1
 *        5:  uniformly distributed on the circle abs(z) = 1
 * 
 * @param[in,out] iseed Seed for the random number generator.
 *      The seed is updated inside the routine ( Currently: seed_out := seed_in + 1 )
 * @param[in] n Length of vector x.
 * @param[out] x Pointer to real vector of length n.
 * 
 * @ingroup auxiliary
 */
template< typename T >
inline void larnv(
    idx_t idist, idx_t* iseed,
    idx_t n, T* x )
{
    typedef real_type<T> real_t;
    using blas::atan;
    using blas::sqrt;
    using blas::cos;
    using blas::sin;
    using blas::internal::set_complex;

    // Constants
    const real_t one   = 1.0;
    const real_t eight = 8.0;
    const real_t twopi = eight * atan(one);

    // Initialize the generator
    std::random_device device;
    std::mt19937 generator(device());
    generator.seed(*iseed);

    if (idist == 1) {
        std::uniform_real_distribution<real_t> d1(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( blas::is_complex<T>::value )
                set_complex(x[i], d1(generator), d1(generator));
            else
                x[i] = d1(generator);
        }
    }
    else if (idist == 2) {
        std::uniform_real_distribution<real_t> d2(-1, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( blas::is_complex<T>::value )
                set_complex(x[i], d2(generator), d2(generator));
            else
                x[i] = d2(generator);
        }
    }
    else if (idist == 3) {
        std::normal_distribution<real_t> d3(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if( blas::is_complex<T>::value )
                set_complex(x[i], d3(generator), d3(generator));
            else
                x[i] = d3(generator);
        }
    }
    else if ( blas::is_complex<T>::value ) {
        if (idist == 4) {
            std::uniform_real_distribution<real_t> d4(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                real_t r     = sqrt(d4(generator));
                real_t theta = twopi * d4(generator);
                set_complex(x[i], r*cos(theta), r*sin(theta));
            }
        }
        else if (idist == 5) {
            std::uniform_real_distribution<real_t> d5(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                real_t theta = twopi * d5(generator);
                set_complex(x[i], cos(theta), sin(theta));
            }
        }
    }

    *iseed = *iseed + 1;
}

}

#endif // __SLATE_LARNV_HH__
