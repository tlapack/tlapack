#ifndef __TLAPACK_CONSTANTS_HH__
#define __TLAPACK_CONSTANTS_HH__

#include <limits>
#include "types.hpp"

namespace blas {

// -----------------------------------------------------------------------------
// Macros to compute scaling constants
//
// __Further details__
//
// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665

// Unit in Last Place
template <typename real_t>
inline const real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

// Safe Minimum such that 1/safe_min() is representable
template <typename real_t>
inline const real_t safe_min()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return max( pow(fradix, expm-1), pow(fradix, 1-expM) );
}

// Safe Maximum such that 1/safe_max() is representable (SAFMAX := 1/SAFMIN)
template <typename real_t>
inline const real_t safe_max()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return min( pow(fradix, 1-expm), pow(fradix, expM-1) );
}

// Safe Minimum such its square is representable
template <typename real_t>
inline const real_t root_min()
{
    return sqrt( safe_min<real_t>() / ulp<real_t>() );
}

// Safe Maximum such that its square is representable
template <typename real_t>
inline const real_t root_max()
{
    return sqrt( safe_max<real_t>() * ulp<real_t>() );
}

}

#endif // __TLAPACK_CONSTANTS_HH__