#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

#include "types.hpp"

namespace blas {

// -----------------------------------------------------------------------------
using std::real;
using std::imag;

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
T max( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
T min( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
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
    static real_t make( real_t re, real_t im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
scalar_t make_scalar( blas::real_type<scalar_t> re,
                      blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename T >
T abs1( T x )
{
    return std::abs( x );
}

template< typename T >
T abs1( std::complex<T> x )
{
    return std::abs( real(x) ) + std::abs( imag(x) );
}

// -----------------------------------------------------------------------------
// is nan
template< typename T >
bool isnan( T x )
{
    return x != x;
}

// -----------------------------------------------------------------------------
// is inf
template< typename T >
bool isinf( T x )
{
    return std::isinf(x);
}

template< typename T >
bool isinf( std::complex<T> x )
{
    return std::isinf( real(x) ) || std::isinf( imag(x) );
}

} // namespace blas

#endif // __TLAPACK_UTILS_HH__