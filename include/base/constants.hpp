// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_CONSTANTS_HH
#define TLAPACK_CONSTANTS_HH

#include <type_traits>
#include <limits>
#include "base/utils.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Macros to compute scaling constants
//
// @details
//
// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665

/** Unit in the Last Place
 * \[
 *      b^{-(p-1)}
 * \]
 * where b is the machine base, and p is the number of digits in the mantissa.
 * 
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

/** Unit roundoff
 * \[
 *      b^{-(p-1)} * r
 * \]
 * where b is the machine base, p is the number of digits in the mantissa,
 * and r is the largest possible rounding error in ULPs (units in the last place)
 * as defined by ISO 10967, which can vary from 0.5 (rounding to the nearest digit)
 * to 1.0 (rounding to zero or to infinity).
 * 
 * @see https://people.eecs.berkeley.edu/~demmel/cs267/lecture21/lecture21.html
 * 
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t uroundoff()
{
    return ulp<real_t>() * std::numeric_limits< real_t >::round_error();
}

/** Digits
 * @ingroup utils
 */
template <typename real_t>
inline const int digits()
{
    return std::numeric_limits< real_t >::digits;
}

/** Safe Minimum such that 1/safe_min() is representable
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t safe_min()
{
    const real_t one( 1 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return max( pow(fradix, expm-one), pow(fradix, one-expM) );
}

/** Safe Maximum such that 1/safe_max() is representable 
 *
 * safe_max() := 1/SAFMIN
 * 
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t safe_max()
{
    const real_t one( 1 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return min( pow(fradix, one-expm), pow(fradix, expM-one) );
}

/** Safe Minimum such its square is representable
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t root_min()
{
    return sqrt( safe_min<real_t>() / ulp<real_t>() );
}

/** Safe Maximum such its square is representable
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t root_max()
{
    return sqrt( safe_max<real_t>() * ulp<real_t>() );
}

/** Blue's min constant b for the sum of squares
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t blue_min()
{
    const real_t half( 0.5 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm   = std::numeric_limits<real_t>::min_exponent;

    return pow( fradix, ceil( half*(expm-1) ) );
}

/** Blue's max constant B for the sum of squares
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t blue_max()
{
    const real_t half( 0.5 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expM   = std::numeric_limits<real_t>::max_exponent;
    const int t      = digits<real_t>();

    return pow( fradix, floor( half*( expM - t + 1 ) ) );
}

/** Blue's scaling constant for numbers smaller than b
 * 
 * @details Modification introduced in @see https://doi.org/10.1145/3061665
 *          to scale denormalized numbers correctly.
 * 
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t blue_scalingMin()
{
    const real_t half( 0.5 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm   = std::numeric_limits<real_t>::min_exponent;
    const int t      = digits<real_t>();

    return pow( fradix, -floor( half*(expm-t) ) );
}

/** Blue's scaling constant for numbers bigger than B
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t blue_scalingMax()
{
    const real_t half( 0.5 );
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expM   = std::numeric_limits<real_t>::max_exponent;
    const int t      = digits<real_t>();

    return pow( fradix, -ceil( half*( expM + t - 1 ) ) );
}

} // namespace tlapack

#endif // TLAPACK_CONSTANTS_HH
