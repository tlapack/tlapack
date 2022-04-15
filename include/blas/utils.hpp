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
using std::sqrt;
using std::sin;
using std::cos;
using std::atan;
using std::exp;
using std::pow;

// -----------------------------------------------------------------------------
// Use MPFR interface
#ifdef USE_MPFR
    inline mpfr::mpreal real( const mpfr::mpreal& x ) { return x; }
    inline mpfr::mpreal imag( const mpfr::mpreal& x ) { return 0; }

    // Argument-dependent lookup (ADL) will include the remaining functions,
    // e.g., mpfr::sin, mpfr::cos.
    // Including them here may cause ambiguous call of overloaded function.
    // See: https://en.cppreference.com/w/cpp/language/adl
#endif

/** Extend conj to real datatypes.
 * 
 * Usage:
 * 
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

template< class T1, class T2, class... Ts >
constexpr bool has_compatible_layout = 
    has_compatible_layout<T1, T2> &&
    has_compatible_layout<T1, Ts...> &&
    has_compatible_layout<T2, Ts...>;

template< class T1, class T2 >
constexpr bool has_compatible_layout<T1,T2> = ( 
    is_same_v< layout_type<T1>, void > ||
    is_same_v< layout_type<T2>, void > ||
    is_same_v< layout_type<T1>, layout_type<T2> >
);

/// Specify the rules for allow_optblas for multiple data structures.
template< class T1, class T2, class... Ts >
struct allow_optblas< T1, T2, Ts... > {
    
    using type = allow_optblas_t<T1>;
    
    static constexpr bool value = 
        allow_optblas_v<T1> &&
        allow_optblas_v<T2, Ts...> &&
        is_same_v< real_type<type>, real_type<allow_optblas_t<T2>> > &&
        has_compatible_layout< T1, T2, Ts... >;
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



// -----------------------------------------------------------------------------
// is_base_of_v is defined in C++17; here's a C++11 definition
#if __cplusplus >= 201703L
    using std::is_base_of_v;
#else
    template< class Base, class Derived >
    constexpr bool is_base_of_v = std::is_base_of<Base,Derived>::value;
#endif

/**
 * @brief Check if a given access type is compatible with the access policy.
 * 
 * Examples of outputs:
 * 
 *      access_granted( MatrixAccessPolicy::UpperTriangle,
 *                      MatrixAccessPolicy::Dense )             returns true.
 *      access_granted( MatrixAccessPolicy::Dense,
 *                      MatrixAccessPolicy::LowerHessenberg )   returns false.
 *      access_granted( Uplo::Upper,
 *                      MatrixAccessPolicy::UpperHessenberg )   returns true.
 *      access_granted( Uplo::Upper,
 *                      Uplo::Lower )                           returns false.
 *      access_granted( upperTriangle,
 *                      Uplo::Lower )                           returns false.
 *      access_granted( strictUpper,
 *                      Uplo::Upper )                           returns true.
 * 
 * @tparam access_t         Access type.
 *      Either MatrixAccessPolicy, Uplo, or any type that implements
 *          operator MatrixAccessPolicy().
 * @tparam accessPolicy_t   Access policy.
 *      Either MatrixAccessPolicy, Uplo, or any type that implements
 *          operator MatrixAccessPolicy().
 * 
 * @param a Access type.
 * @param p Access policy.
 * 
 * @ingroup utils
 */
template< class access_t, class accessPolicy_t >
inline constexpr
bool access_granted( access_t a, accessPolicy_t p )
{
    return (
        
        is_base_of_v< accessPolicy_t, access_t > ||

        ((MatrixAccessPolicy) p ==(MatrixAccessPolicy) a) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::Dense) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::UpperHessenberg &&
        (
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::UpperTriangle) || 
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::StrictUpper)
        )) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::LowerHessenberg &&
        (
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::LowerTriangle) || 
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::StrictUpper)
        )) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::UpperTriangle &&
        (
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::StrictUpper)
        )) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::LowerTriangle &&
        (
            ((MatrixAccessPolicy) a == MatrixAccessPolicy::StrictUpper)
        ))
    );
}

/**
 * @brief Check if a given access type is compatible with the access policy.
 * 
 * Specific implementation for band_t.
 * 
 * @see bool access_granted( access_t a, accessPolicy_t p )
 * 
 * @ingroup utils
 */
template< class accessPolicy_t >
inline constexpr
bool access_granted( band_t a, accessPolicy_t p )
{
    return (
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::Dense) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::UpperHessenberg && a.lower_bandwidth <= 1) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::LowerHessenberg && a.upper_bandwidth <= 1) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::UpperTriangle && a.lower_bandwidth == 0) ||
        ((MatrixAccessPolicy) p == MatrixAccessPolicy::LowerTriangle && a.upper_bandwidth == 0)
    );
}

/**
 * @brief Check if a given access type is compatible with the access policy.
 * 
 * Specific implementation for band_t.
 * 
 * @see bool access_granted( access_t a, accessPolicy_t p )
 * 
 * @ingroup utils
 */
template< class access_t >
inline constexpr
bool access_granted( access_t a, band_t p )
{
    return false;
}

/**
 * @brief Check if a given access type is compatible with the access policy.
 * 
 * Specific implementation for band_t.
 * 
 * @see bool access_granted( access_t a, accessPolicy_t p )
 * 
 * @ingroup utils
 */
inline constexpr
bool access_granted( band_t a, band_t p )
{
    return  (p.lower_bandwidth >= a.lower_bandwidth) &&
            (p.upper_bandwidth >= a.upper_bandwidth);
}

/**
 * @return ! access_granted( a, p ).
 * 
 * @see bool access_granted( access_t a, accessPolicy_t p )
 * 
 * @ingroup utils
 */
template< class access_t, class accessPolicy_t >
inline constexpr
bool access_denied( access_t a, accessPolicy_t p ) {
    return ! access_granted( a, p );
}

} // namespace blas

#endif // __TBLAS_UTILS_HH__
