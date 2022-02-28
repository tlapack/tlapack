// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_UTILS_HH__
#define __TBLAS_UTILS_HH__

#include "blas/types.hpp"
#include "blas/exceptionHandling.hpp"

#include <limits>
#include <cmath>
#include <utility>

#ifdef USE_MPFR
    #include <mpreal.h>
#endif

namespace blas {

// -----------------------------------------------------------------------------
// enable_if_t is defined in C++14; here's a C++11 definition
#if __cplusplus >= 201402L
    using std::enable_if_t;
#else
    template< bool B, class T = void >
    using enable_if_t = typename enable_if<B,T>::type;
#endif

// -----------------------------------------------------------------------------
// is_same_v is defined in C++17; here's a C++11 definition
#if __cplusplus >= 201703L
    using std::is_same_v;
#else
    template< class T, class U >
    constexpr bool is_same_v = std::is_same<T, U>::value;
#endif

// -----------------------------------------------------------------------------
// Use routines from std C++
using std::real;
using std::imag;
using std::isinf;
using std::isnan;
using std::ceil;
using std::floor;

// -----------------------------------------------------------------------------
// Use MPFR interface
#ifdef USE_MPFR
    inline mpfr::mpreal real( const mpfr::mpreal& x ) { return x; }
    inline mpfr::mpreal imag( const mpfr::mpreal& x ) { return 0; }
    using mpfr::isinf;
    using mpfr::isnan;
    using mpfr::ceil;
    using mpfr::floor;
#endif

/** Extend conj to real datatypes.
 * 
 * @usage:
 *     using blas::conj;
 *     scalar_t x = ...
 *     scalar_t y = conj( x );
 * 
 * @param[in] x Real number
 * @return x
 * 
 * @note C++11 to C++17 returns complex<real_t> instead of real_t. @see std::conj
 * 
 * @ingroup utils
 */
template< typename real_t >
inline real_t conj( const real_t& x )
{
    // This prohibits complex types; it can't be called as y = blas::conj( x ).
    static_assert( ! is_complex<real_t>::value,
                    "Usage: using blas::conj; y = conj(x); NOT: y = blas::conj(x);" );
    return x;
}

// -----------------------------------------------------------------------------
// max that works with different data types
// and any number of arguments: max( a, b, c, d )

// one argument
template< typename T >
inline T max( const T& x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    max( const T1& x, const T2& y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    max( const T1& first, const Types&... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
inline T min( const T& x ) { return x; }

// two arguments
template< typename T1, typename T2 >
inline scalar_type< T1, T2 >
    min( const T1& x, const T2& y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
inline scalar_type< T1, Types... >
    min( const T1& first, const Types&... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static inline real_t make( const real_t& re, const real_t& im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static inline std::complex<real_t> make( const real_t& re, const real_t& im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
inline scalar_t make_scalar( blas::real_type<scalar_t> re,
                             blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
/// Type-safe sgn function
/// @see Source: https://stackoverflow.com/a/4609795/5253097
///
template <typename real_t>
inline int sgn( const real_t& val ) {
    return (real_t(0) < val) - (val < real_t(0));
}

#ifdef USE_MPFR
    template<> 
    inline int sgn( const mpfr::mpreal& x )
    { return mpfr::sgn( x ); }
#endif

// -----------------------------------------------------------------------------
/// sqrt, needed because std C++ template returns double for any type different
/// float anf long double.
/// Note that the template in std::complex return the desired std::complex<T>.
template< typename T >
inline T sqrt( const T& x )
{ return std::sqrt( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal sqrt( const mpfr::mpreal& x )
    { return mpfr::sqrt( x ); }
#endif

// -----------------------------------------------------------------------------
/// sin, needed because std C++11 template returns double.
template< typename T >
inline T sin( const T& x ) { return std::sin( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal sin( const mpfr::mpreal& x ) { return mpfr::sin( x ); }
#endif

// -----------------------------------------------------------------------------
/// cos, needed because std C++11 template returns double.
template< typename T >
inline T cos( const T& x ) { return std::cos( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal cos( const mpfr::mpreal& x ) { return mpfr::cos( x ); }
#endif

// -----------------------------------------------------------------------------
/// atan, needed because std C++ template returns double.
template< typename T >
inline T atan( const T& x ) { return std::atan( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal atan( const mpfr::mpreal& x ) { return mpfr::atan( x ); }
#endif

// -----------------------------------------------------------------------------
/// exp, needed because std C++ template returns double.
template< typename T >
inline T exp( const T& x ) { return std::exp( x ); }

#ifdef USE_MPFR
    template<> 
    inline mpfr::mpreal exp( const mpfr::mpreal& x ) { return mpfr::exp( x ); }
#endif

// -----------------------------------------------------------------------------
/// pow, avoids promotion to double from std C++.
/// Note that the template in std::complex return the desired std::complex<T>.
template< typename T >
inline T pow( const T& base, const T& exp )
{ return std::pow( base, exp ); }

template< typename T >
inline T pow( const int base, const T& exp )
{ return std::pow( (double)base, exp ); }

#ifdef USE_MPFR
    template<>
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const unsigned int b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const int b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const long double b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const mpfr::mpreal& a, const double b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const unsigned int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const long int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    template<>
    inline mpfr::mpreal pow(const int a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const long double a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
    inline mpfr::mpreal pow(const double a, const mpfr::mpreal& b)
    {
        return mpfr::pow( a, b );
    }
#endif

// -----------------------------------------------------------------------------
/// isnan for complex numbers
template< typename real_t >
inline bool isnan( const std::complex<real_t>& x )
{
    return isnan( real(x) ) || isnan( imag(x) );
}

// -----------------------------------------------------------------------------
/// isinf for complex numbers
template< typename real_t >
inline bool isinf( const std::complex<real_t>& x )
{
    return isinf( real(x) ) || isinf( imag(x) );
}

// -----------------------------------------------------------------------------
/// 2-norm absolute value, sqrt( |Re(x)|^2 + |Im(x)|^2 )
///
/// Note that std::abs< std::complex > does not overflow or underflow at
/// intermediate stages of the computation.
/// @see https://en.cppreference.com/w/cpp/numeric/complex/abs
/// but it may not propagate NaNs.
///
template< typename T >
inline real_type<T> abs( const T& x, bool check = true ) {
    if( is_complex<T>::value && check ) {
        if( isnan(x) )
            return std::numeric_limits< real_type<T> >::quiet_NaN();
    }
    return std::abs( x ); // Contains the 2-norm for the complex case
}

#ifdef USE_MPFR
    /// Absolute value
    template<>
    inline mpfr::mpreal abs( const mpfr::mpreal& x, bool check ) {
        return mpfr::abs( x );
    }
    
    /// 2-norm absolute value, sqrt( |Re(x)|^2 + |Im(x)|^2 )
    ///
    /// Note that std::abs< mpfr::mpreal > may not propagate Infs.
    ///
    template<>
    inline mpfr::mpreal abs( const std::complex<mpfr::mpreal>& x, bool check ) {
        if( check ) {
            if( isnan(x) )
                return std::numeric_limits< mpfr::mpreal >::quiet_NaN();
            else if( isinf(x) )
                return std::numeric_limits< mpfr::mpreal >::infinity();
        }
        return std::abs( x );
    }
#endif

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename real_t >
inline real_t abs1( const real_t& x ) { return abs( x, false ); }

template< typename real_t >
inline real_t abs1( const std::complex<real_t>& x )
{
    return abs( real(x), false ) + abs( imag(x), false );
}

// -----------------------------------------------------------------------------
/// Optimized BLAS

/// Specify the rules for allow_optblas for multiple data structures.
template< class T1, class T2, class... Ts >
struct allow_optblas< T1, T2, Ts... > {
    
    using type = allow_optblas_t<T1>;
    
    static constexpr Layout layout =
        (   allow_optblas_l<T1> == Layout::Scalar || 
            allow_optblas_l<T1> == Layout::StridedVector )
                ? allow_optblas_l<T2, Ts...>
                : allow_optblas_l<T1>;
    
    static constexpr bool value = 
        allow_optblas_v<T1> &&
        allow_optblas_v<T2, Ts...> &&
        is_same_v< real_type<type>, real_type<allow_optblas_t<T2>> > &&
        (   allow_optblas_l<T1> == Layout::Scalar || 
            allow_optblas_l<T1> == Layout::StridedVector ||
            allow_optblas_l<T2, Ts...> == Layout::Scalar ||
            allow_optblas_l<T2, Ts...> == Layout::StridedVector ||
            layout == allow_optblas_l<T2, Ts...> );
};

template<class T1, class... Ts>
using enable_if_allow_optblas_t = enable_if_t<(
    allow_optblas_v< T1, Ts... >
), int >;

template<class T1, class... Ts>
using disable_if_allow_optblas_t = enable_if_t<(
    ! allow_optblas_v< T1, Ts... >
), int >;

#define TLAPACK_OPT_TYPE( T ) \
    template<> struct allow_optblas< T > { \
        using type = T; \
        static constexpr Layout layout = Layout::Scalar; \
        static constexpr bool value = true; \
    }

    /// Optimized types
    #ifdef TLAPACK_USE_OPTSINGLE
        TLAPACK_OPT_TYPE(float);
    #endif
    #ifdef TLAPACK_USE_OPTDOUBLE
        TLAPACK_OPT_TYPE(double);
    #endif
    #ifdef TLAPACK_USE_OPTCOMPLEX
        TLAPACK_OPT_TYPE(std::complex<float>);
    #endif
    #ifdef TLAPACK_USE_OPTDOUBLECOMPLEX
        TLAPACK_OPT_TYPE(std::complex<double>);
    #endif
#undef TLAPACK_OPT_TYPE

} // namespace blas

#endif // __TBLAS_UTILS_HH__
