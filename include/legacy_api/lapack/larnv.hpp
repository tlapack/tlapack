/// @file larnv.hpp Returns a vector of random numbers from a uniform or normal distribution.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larnv.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LARNV_HH
#define TLAPACK_LEGACY_LARNV_HH

#include "legacy_api/base/types.hpp"
#include "lapack/larnv.hpp"

namespace tlapack {

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
    using internal::vector;
    auto x_ = vector( x, n );

    if (idist == 1) return larnv<1>( *iseed, x_ );
    else if (idist == 2) return larnv<2>( *iseed, x_ );
    else if (idist == 3) return larnv<3>( *iseed, x_ );
    else if (idist == 4) return larnv<4>( *iseed, x_ );
    else if (idist == 5) return larnv<5>( *iseed, x_ );
}

}

#endif // TLAPACK_LEGACY_LARNV_HH
