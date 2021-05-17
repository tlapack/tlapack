#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

#include "blas/types.hpp"
#include <limits>

namespace blas {

// -----------------------------------------------------------------------------
// Use routines from std C++
using std::real;
using std::imag;
using std::abs; // Contains the 2-norm for the complex case
using std::isinf;
using std::isnan;

// -----------------------------------------------------------------------------
// Use MPFR interface
#ifdef USE_GNU_MPFR
    inline mpfr::mpreal real( const mpfr::mpreal& x ) { return x; }
    inline mpfr::mpreal imag( const mpfr::mpreal& x ) { return 0; }
    using mpfr::abs;
    using mpfr::isinf;
    using mpfr::isnan;
#endif

/// Extend conj to real datatypes.
/// For real T, this returns type T, whereas C++11 returns complex<T>.
/// Usage:
///     using blas::conj;
///     scalar_t x = ...
///     scalar_t y = conj( x );
/// That will use std::conj for complex types, and blas::conj for other types.
/// This prohibits complex types; it can't be called as y = blas::conj( x ).
///
template< typename T >
inline T conj( T x )
{
    static_assert(
        ! is_complex<T>::value,
        "Usage: using blas::conj; y = conj(x); NOT: y = blas::conj(x);" );
    return x;
}

// -----------------------------------------------------------------------------
// max that works with different data types
// and any number of arguments: max( a, b, c, d )

// one argument
template< typename T >
inline T max( T x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
inline T min( T x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    min( T1 first, Types... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static inline real_t make( real_t re, real_t im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static inline std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
inline scalar_t make_scalar( blas::real_type<scalar_t> re,
                             blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
// sqrt, needed because std C++11 template returns double
// In the MPFR, the template returns mpfr::mpreal
// Note that the template in std::complex return the desired std::complex<T>
template< typename T >
inline T sqrt( const T& x )
{
#ifdef USE_GNU_MPFR
    return T( mpfr::sqrt( x ) );
#else
    return T( std::sqrt( x ) );
#endif
}

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename T >
inline T abs1( T x ) { return abs( x ); }

template< typename T >
inline T abs1( std::complex<T> x )
{
    return abs( real(x) ) + abs( imag(x) );
}

// -----------------------------------------------------------------------------
// isnan for complex numbers
template< typename T >
inline bool isnan( std::complex<T> x )
{
    return isnan( real(x) ) || isnan( imag(x) );
}

// -----------------------------------------------------------------------------
// isinf for complex numbers
template< typename T >
inline bool isinf( std::complex<T> x )
{
    return isinf( real(x) ) || isinf( imag(x) );
}

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

} // namespace blas

#endif // __TLAPACK_UTILS_HH__