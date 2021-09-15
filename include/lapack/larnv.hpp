/// @file larnv.hpp Returns a vector of random numbers from a uniform or normal distribution.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larnv.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARNV_HH__
#define __LARNV_HH__

#include <random>
#include "lapack/types.hpp"

namespace lapack {

/**
 * @brief Returns a vector of n random real numbers from a uniform or normal distribution.
 * 
 * This implementation uses the Mersenne Twister 19937 generator (class std::mt19937),
 * which is a Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
 * 
 * Requires ISO C++ 2011 random number generators.
 * 
 * @param[in] idist Specifies the distribution:
 *
 *        1: uniform (0,1)
 *        2: uniform (-1,1)
 *        3: normal (0,1)
 * 
 * @param[in,out] iseed Integer array
 *          On entry, the seed of the random number generator; the array
 *          elements must be between 0 and 4095, and iseed[4] must be odd.
 *          On exit, the seed is updated.
 * @param[in] n Length of vector x.
 * @param[out] x Pointer to real vector of length n.
 * 
 * @ingroup auxiliary
 */
template< typename real_t >
void larnv(
    blas::idx_t idist, blas::idx_t* iseed,
    blas::idx_t n, real_t* x )
{
    // Initialize the generator
    std::random_device device;
    std::mt19937 generator(device());
    generator.seed(iseed[0]); // TODO: fix me

    // uniform (0,1)
    if (idist == 1) {
        std::uniform_real_distribution<real_t> d1(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = d1(generator);
    }
    // uniform (-1,1)
    else if (idist == 2) {
        std::uniform_real_distribution<real_t> d2(-1, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = d2(generator);
    }
    // normal (0,1)
    else if (idist == 3) {
        std::normal_distribution<real_t> d3(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = d3(generator);
    }

    iseed[0]++; // TODO: fix me
}

/**
 * @brief Returns a vector of n random complex numbers from a uniform or normal distribution.
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
 * @param[in] iseed Integer array
 *          On entry, the seed of the random number generator; the array
 *          elements must be between 0 and 4095, and iseed[4] must be odd.
 *          On exit, the seed is updated.
 * @param[in] n Length of vector x.
 * @param[out] x Pointer to real vector of length n.
 * 
 * @ingroup auxiliary
 */
template< typename T >
void larnv(
    blas::idx_t idist, blas::idx_t* iseed,
    blas::idx_t n, blas::complex_type<T>* x )
{
    typedef real_type<T> real_t;
    typedef complex_type<T> complex_t;
    using blas::atan;
    using blas::sqrt;
    using blas::exp;

    // Constants
    const real_t zero  = 0.0;
    const real_t one   = 1.0;
    const real_t eight = 8.0;
    const real_t twopi = eight * atan(one);

    // Initialize the generator
    std::random_device device;
    std::mt19937 generator(device());
    generator.seed(iseed[0]); // TODO: fix me

    if (idist == 1) {
        std::uniform_real_distribution<real_t> d1(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = complex_t(d1(generator), d1(generator));
    }
    else if (idist == 2) {
        std::uniform_real_distribution<real_t> d2(-1, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = complex_t(d2(generator), d2(generator));
    }
    else if (idist == 3) {
        std::normal_distribution<real_t> d3(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = complex_t(d3(generator), d3(generator));
    }
    else if (idist == 4) {
        std::uniform_real_distribution<real_t> d4(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = sqrt(d4(generator)) * exp(complex_t(zero, twopi * d4(generator)));
    }
    else if (idist == 5) {
        std::uniform_real_distribution<real_t> d5(0, 1);
        for (idx_t i = 0; i < n; ++i)
            x[i] = exp(complex_t(zero, twopi * d5(generator)));
    }

    iseed[0]++; // TODO: fix me
}

}

#endif // __LARNV_HH__
