// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_CONSTANTS_HH__
#define __TBLAS_CONSTANTS_HH__

#include <type_traits>
#include <limits>
#include "blas/utils.hpp"

#ifdef USE_MPFR
    #include <mpreal.h>
#endif

namespace blas {

/// INVALID_INDEX = ( std::is_unsigned< blas::size_t >::value )
///               ? std::numeric_limits< blas::size_t >::max()
///               : -1;                                       
const blas::size_t INVALID_INDEX( -1 );

// -----------------------------------------------------------------------------
// Macros to compute scaling constants
//
// __Further details__
//
// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665

/// Unit in Last Place

/** Unit in Last Place
 * @ingroup utils
 */
template <typename real_t>
inline constexpr real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

/** Digits
 * @ingroup utils
 */
template <typename real_t>
inline const real_t digits()
{
    return std::numeric_limits< real_t >::digits;
}

#ifdef USE_MPFR
    namespace internal {
        const mpfr::mpreal mpreal_digits = std::numeric_limits< mpfr::mpreal >::digits();
    }
    #ifdef MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS
        /** Digits for the mpfr::mpreal datatype
         * @ingroup utils
         */
        template<> inline const mpfr::mpreal digits() {
            return internal::mpreal_digits; 
        }
    #else
        /** Digits for the mpfr::mpreal datatype
         * @ingroup utils
         */
        template<> inline const mpfr::mpreal digits() {
            return std::numeric_limits< mpfr::mpreal >::digits;
        }
    #endif
#endif

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

} // namespace blas

#endif // __TBLAS_CONSTANTS_HH__
