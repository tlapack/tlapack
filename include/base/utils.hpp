// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

#include <limits>
#include <cmath>
#include <utility>
#include <type_traits>

#include "base/types.hpp"
#include "base/arrayTraits.hpp"
#include "base/exceptionHandling.hpp"

// -----------------------------------------------------------------------------
// Macros to handle error checks
#if defined(TLAPACK_ERROR_NDEBUG) || defined(NDEBUG)

    // <T>BLAS does no error checking;
    // lower level BLAS may still handle errors via xerbla
    #define lapack_error( msg, code ) \
        ((void)0)
    #define lapack_error_if( cond, code ) \
        ((void)0)

#else

    /// internal macro to the get string __func__
    /// ex: lapack_error( "a < b", -2 );
    /// @returns code
    #define lapack_error( msg, code ) do { \
        tlapack::error( msg, __func__ ); \
        return code; \
    } while(false)

    /// internal macro to get strings: #cond and __func__
    /// ex: lapack_error_if( a < b, -6 );
    /// @returns code if a < b
    #define lapack_error_if( cond, code ) do { \
        if( cond ) { \
            tlapack::error( #cond, __func__ ); \
            return code; \
        } \
    } while(false)

#endif

namespace tlapack {

// -----------------------------------------------------------------------------
// Use routines from std C++
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
using std::pair;

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

//------------------------------------------------------------------------------
/// True if T is std::complex<T2> for some type T2.
template <typename T>
struct is_complex:
    std::integral_constant<bool, false>
{};

/// specialize for std::complex
template <typename T>
struct is_complex< std::complex<T> >:
    std::integral_constant<bool, true>
{};

template <typename T, enable_if_t<!is_complex<T>::value,int> = 0>
inline constexpr
real_type<T> real( const T& x ) { return x; }

template <typename T, enable_if_t<!is_complex<T>::value,int> = 0>
inline constexpr
real_type<T> imag( const T& x ) { return 0; }

/** Extend conj to real datatypes.
 * 
 * Usage:
 * 
 *     using tlapack::conj;
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
    // This prohibits complex types; it can't be called as y = tlapack::conj( x ).
    static_assert( ! is_complex<real_t>::value,
                    "Usage: using tlapack::conj; y = conj(x); NOT: y = tlapack::conj(x);" );
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
inline scalar_t make_scalar( real_type<scalar_t> re,
                             real_type<scalar_t> im=0 )
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
inline auto abs( const T& x ) {
    if( is_complex<T>::value ) {
        if( isnan(x) )
            return std::numeric_limits< real_type<T> >::quiet_NaN();
    }
    return std::abs( x ); // Contains the 2-norm for the complex case
}

// -----------------------------------------------------------------------------
/// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename real_t >
inline real_t abs1( const real_t& x ) { return abs( x ); }

template< typename real_t >
inline real_t abs1( const std::complex<real_t>& x )
{
    return abs( real(x) ) + abs( imag(x) );
}

// -----------------------------------------------------------------------------
/// Optimized BLAS

template< class P1, class P2, class... Ps >
constexpr bool has_compatible_layout = 
    has_compatible_layout<P1, P2> &&
    has_compatible_layout<P1, Ps...> &&
    has_compatible_layout<P2, Ps...>;

template< class C1, class C2 >
constexpr bool has_compatible_layout< C1, C2 > = (
    ( layout<C1> == Layout::Unspecified ) ||
    ( layout<C2> == Layout::Unspecified ) ||
    ( layout<C1> == layout<C2> )
);

template< class C1, class T1, class C2, class T2 >
constexpr bool has_compatible_layout< pair<C1,T1>, pair<C2,T2> > =
    has_compatible_layout< C1, C2 >;

template< class C, class T >
struct allow_optblas< pair<C,T> > {
    static constexpr bool value = 
        allow_optblas_v<C> &&
        allow_optblas_v<T> &&
        is_same_v< type_t<C>, T >;
};

template< class P1, class P2, class... Ps >
struct allow_optblas< P1, P2, Ps... > {
    static constexpr bool value = 
        allow_optblas_v< P1 > &&
        allow_optblas_v< P2, Ps... > &&
        has_compatible_layout< P1, P2, Ps... >;
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
        static constexpr bool value = true; \
    }; \
    template<> struct type_trait< T > { \
        using type = T; \
    }

    /// Optimized types
    #ifdef USE_BLASPP_WRAPPERS
        TLAPACK_OPT_TYPE(float);
        TLAPACK_OPT_TYPE(double);
        TLAPACK_OPT_TYPE(std::complex<float>);
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

/** Defines structures that check if opts_t has an attribute called member.
 * 
 * has_member<opts_t> is a true type if the struct opts_t has an attribute called member.
 * has_member_v<opts_t> is true if the struct opts_t has a member called member.
 * 
 * @param member Attribute to be checked
 * 
 * @see https://stackoverflow.com/questions/1005476/how-to-detect-whether-there-is-a-specific-member-variable-in-class
 */
#define TLAPACK_MEMBER_CHECKER(member) \
    template< class opts_t, typename = int > \
    struct has_ ## member : std::false_type { }; \
    \
    template< class opts_t > \
    struct has_ ## member< opts_t, \
        enable_if_t< \
            !is_same_v< \
                decltype(std::declval<opts_t>().member), \
                void > \
        , int > \
    > : std::true_type { }; \
    \
    template< class opts_t > \
    constexpr bool has_ ## member ## _v = has_ ## member<opts_t>::value;

// Activate checkers:
TLAPACK_MEMBER_CHECKER(nb)
TLAPACK_MEMBER_CHECKER(workPtr)

/** has_work_v<opts_t> is true if the struct opts_t has a member called workPtr.
 */
template< class opts_t > constexpr bool has_work_v = has_workPtr_v<opts_t>;

/**
 * @return a default value if nb is not a member of opts_t.
 */
template< class opts_t, enable_if_t< !has_nb_v<opts_t>, int > = 0 >
inline constexpr auto get_nb( opts_t&& opts ) {
    /// TODO: Put default values somewhere else
    return 32;
}

/**
 * @return opts.nb if nb is a member of opts_t and opts.nb > 0.
 * @return a default value otherwise.
 */
template< class opts_t, enable_if_t<  has_nb_v<opts_t>, int > = 0 >
inline constexpr auto get_nb( opts_t&& opts )
-> std::remove_reference_t<decltype(opts.nb)> {
    return ( opts.nb > 0 ) 
        ? opts.nb
        : get_nb( 0 ); // get default nb
}

/**
 * @return *(opts.workPtr) if workPtr is a member of opts_t.
 */
template< class opts_t, enable_if_t<  has_workPtr_v<opts_t>, int > = 0 >
inline constexpr auto get_work( opts_t&& opts ) {
    if ( opts.workPtr )
        return *(opts.workPtr);
    else {    
        /// TODO: Allocate space.
        /// TODO: Create matrix.
        /// TODO: Return matrix.
        return *(opts.workPtr);
    }
}

} // namespace tlapack

#endif // __TLAPACK_UTILS_HH__