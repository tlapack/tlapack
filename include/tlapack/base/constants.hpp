/// @file constants.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_CONSTANTS_HH
#define TLAPACK_CONSTANTS_HH

#include <limits>
#include <type_traits>

#include "tlapack/base/utils.hpp"

namespace tlapack {

// -----------------------------------------------------------------------------
// Macros to compute scaling constants

/** Unit in the Last Place
 * \[
 *      b^{-(p-1)}
 * \]
 * where b is the machine base, and p is the number of digits in the mantissa.
 *
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t ulp() noexcept
{
    return std::numeric_limits<real_t>::epsilon();
}

/** Unit roundoff
 * \[
 *      b^{-(p-1)} * r
 * \]
 * where b is the machine base, p is the number of digits in the mantissa,
 * and r is the largest possible rounding error in ULPs (units in the last
 * place) as defined by ISO 10967, which can vary from 0.5 (rounding to the
 * nearest digit) to 1.0 (rounding to zero or to infinity).
 *
 * @see https://people.eecs.berkeley.edu/~demmel/cs267/lecture21/lecture21.html
 *
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t uroundoff() noexcept
{
    return ulp<real_t>() * std::numeric_limits<real_t>::round_error();
}

/** Digits
 *
 * Number of digits p in the mantissa.
 * @see std::numeric_limits<real_t>::digits.
 *
 * @ingroup constants
 */
template <typename real_t>
int digits() noexcept
{
    return std::numeric_limits<real_t>::digits;
}

/** Safe Minimum
 *
 * Smallest normal positive power of two such that its inverse (1/safe_min()) is
 * finite.
 *
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t safe_min() noexcept
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return pow(fradix, real_t(max(expm - 1, 1 - expM)));
}

/** Safe Maximum
 *
 * safe_max() := 1/safe_min()
 *
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t safe_max() noexcept
{
    return real_t(1) / safe_min<real_t>();
}

/** Blue's min constant b for the sum of squares
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t blue_min() noexcept
{
    const real_t half(0.5);
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;

    return pow(fradix, ceil(half * real_t(expm - 1)));
}

/** Blue's max constant B for the sum of squares
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t blue_max() noexcept
{
    const real_t half(0.5);
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expM = std::numeric_limits<real_t>::max_exponent;
    const int t = digits<real_t>();

    return pow(fradix, floor(half * real_t(expM - t + 1)));
}

/** Blue's scaling constant for numbers smaller than b
 * @see https://doi.org/10.1145/355769.355771
 *
 * @details Modification introduced in @see https://doi.org/10.1145/3061665
 *          to scale denormalized numbers correctly.
 *
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t blue_scalingMin() noexcept
{
    const real_t half(0.5);
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int t = digits<real_t>();

    return pow(fradix, -floor(half * real_t(expm - t)));
}

/** Blue's scaling constant for numbers bigger than B
 * @see https://doi.org/10.1145/355769.355771
 * @ingroup constants
 */
template <TLAPACK_REAL real_t>
constexpr real_t blue_scalingMax() noexcept
{
    const real_t half(0.5);
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expM = std::numeric_limits<real_t>::max_exponent;
    const int t = digits<real_t>();

    return pow(fradix, -ceil(half * real_t(expM + t - 1)));
}

}  // namespace tlapack

#endif  // TLAPACK_CONSTANTS_HH
