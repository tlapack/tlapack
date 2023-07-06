/// @file larnv.hpp Returns a vector of random numbers from a uniform or normal
/// distribution.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/mastUSA
/// @note Adapted
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
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
template <int idist, TLAPACK_VECTOR vector_t, class Sseq>
void larnv(Sseq& iseed, vector_t& x)
{
    using idx_t = size_type<vector_t>;
    using T = type_t<vector_t>;
    using real_t = real_type<T>;

    // Constants
    const idx_t n = size(x);
    const double twopi(8 * std::atan(1.0));

    // Initialize the Mersenne Twister generator
    std::random_device device;
    std::mt19937 generator(device());
    generator.seed(iseed);

    if constexpr (idist == 1) {
        std::uniform_real_distribution<> d1(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (is_complex<T>)
                x[i] = T(real_t(d1(generator)), real_t(d1(generator)));
            else
                x[i] = T(d1(generator));
        }
    }
    else if constexpr (idist == 2) {
        std::uniform_real_distribution<> d2(-1, 1);
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (is_complex<T>)
                x[i] = T(real_t(d2(generator)), real_t(d2(generator)));
            else
                x[i] = T(d2(generator));
        }
    }
    else if constexpr (idist == 3) {
        std::normal_distribution<> d3(0, 1);
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (is_complex<T>)
                x[i] = T(real_t(d3(generator)), real_t(d3(generator)));
            else
                x[i] = T(d3(generator));
        }
    }
    else if constexpr (is_complex<T>) {
        if constexpr (idist == 4) {
            std::uniform_real_distribution<> d4(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                double r = sqrt(d4(generator));
                double theta = twopi * d4(generator);
                x[i] = T(r * cos(theta), r * sin(theta));
            }
        }
        else if constexpr (idist == 5) {
            std::uniform_real_distribution<> d5(0, 1);
            for (idx_t i = 0; i < n; ++i) {
                double theta = twopi * d5(generator);
                x[i] = T(real_t(cos(theta)), real_t(sin(theta)));
            }
        }
    }

    // Update the seed
    iseed = iseed + 1;
}

}  // namespace tlapack

#endif  // TLAPACK_LARNV_HH
