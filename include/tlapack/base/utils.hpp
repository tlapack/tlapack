// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UTILS_HH
#define TLAPACK_UTILS_HH

#include <limits>
#include <cmath>
#include <utility>
#include <type_traits>

#include "tlapack/base/types.hpp"
#include "tlapack/base/exceptionHandling.hpp"
#include "tlapack/base/legacyArray.hpp"

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

template<typename idx_t>
using range = pair<idx_t,idx_t>;
using std::enable_if_t;

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
template< typename real_t, enable_if_t<!is_complex<real_t>::value,int> = 0 >
inline constexpr
real_t conj( const real_t& x )
{
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
/// isinf for complex numbers
template< typename real_t >
inline bool isinf( const std::complex<real_t>& x )
{
    return isinf( real(x) ) || isinf( imag(x) );
}

namespace internal {

    template< class array_t, typename = int >
    struct has_operator_parenthesis_with_2_indexes : std::false_type { };

    template< class array_t >
    struct has_operator_parenthesis_with_2_indexes<
        array_t,
        enable_if_t<
            !is_same_v<
                decltype( std::declval<array_t>()(0,0) )
            , void >
        , int >
    > : std::true_type { };

    template< class array_t, typename = int >
    struct has_operator_brackets_with_1_index : std::false_type { };

    template< class array_t >
    struct has_operator_brackets_with_1_index<
        array_t,
        enable_if_t<
            !is_same_v<
                decltype( std::declval<array_t>()[0] )
            , void >
        , int >
    > : std::true_type { };
    
}

template< class array_t >
constexpr bool is_matrix =
    internal::has_operator_parenthesis_with_2_indexes<array_t>::value;

template< class array_t >
constexpr bool is_vector =
    ( !is_matrix<array_t> ) &&
    internal::has_operator_brackets_with_1_index<array_t>::value;

namespace internal {

    /**
     * @brief Data type trait.
     * 
     * The data type is defined on @c type_trait<array_t>::type.
     * 
     * @tparam T A non-array class.
     */
    template< class T, typename = int >
    struct type_trait {
        using type = void;
    };

    /**
     * @brief Data type trait.
     * 
     * The data type is defined on @c type_trait<array_t>::type.
     * 
     * @tparam matrix_t Matrix class.
     */
    template< class matrix_t >
    struct type_trait< matrix_t, enable_if_t< is_matrix< matrix_t >, int > > {
        using type = typename std::decay< decltype( std::declval<matrix_t>()(0,0) ) >::type;
    };

    /**
     * @brief Data type trait.
     * 
     * The data type is defined on @c type_trait<array_t>::type.
     * 
     * @tparam vector_t Vector class.
     */
    template< class vector_t >
    struct type_trait< vector_t, enable_if_t< is_vector< vector_t >, int > > {
        using type = typename std::decay< decltype( std::declval<vector_t>()[0] ) >::type;
    };

    /**
     * @brief Size type trait.
     * 
     * The size type is defined on @c sizet_trait<array_t>::type.
     * 
     * @tparam T A non-array class.
     */
    template< class T, typename = int >
    struct sizet_trait {
        using type = void;
    };

    /**
     * @brief Size type trait.
     * 
     * The size type is defined on @c sizet_trait<array_t>::type.
     * 
     * @tparam matrix_t Matrix class.
     */
    template< class matrix_t >
    struct sizet_trait< matrix_t, enable_if_t< is_matrix< matrix_t >, int > > {
        using type = typename std::decay< decltype( nrows(std::declval<matrix_t>()) ) >::type;
    };

    /**
     * @brief Size type trait.
     * 
     * The size type is defined on @c sizet_trait<array_t>::type.
     * 
     * @tparam vector_t Vector class.
     */
    template< class vector_t >
    struct sizet_trait< vector_t, enable_if_t< is_vector< vector_t >, int > > {
        using type = typename std::decay< decltype( size(std::declval<vector_t>()) ) >::type;
    };

}

/// Alias for @c type_trait<>::type.
template< class array_t >
using type_t = typename internal::type_trait< array_t >::type;

/// Alias for @c sizet_trait<>::type.
template< class array_t >
using size_type = typename internal::sizet_trait< array_t >::type;

/**
 * Returns true if and only if A has an infinite entry.
 * 
 * @tparam access_t Type of access inside the algorithm.
 *      Either MatrixAccessPolicy or any type that implements
 *          operator MatrixAccessPolicy().
 * 
 * @param[in] accessType Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          MatrixAccessPolicy::Dense,
 *          MatrixAccessPolicy::UpperHessenberg,
 *          MatrixAccessPolicy::LowerHessenberg,
 *          MatrixAccessPolicy::UpperTriangle,
 *          MatrixAccessPolicy::LowerTriangle,
 *          MatrixAccessPolicy::StrictUpper,
 *          MatrixAccessPolicy::StrictLower.
 * 
 * @param[in] A matrix.
 * 
 * @return true if A has an infinite entry.
 * @return false if A has no infinite entry.
 */
template< class access_t, class matrix_t >
bool hasinf( access_t accessType, const matrix_t& A ) {

    using idx_t  = size_type< matrix_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::UpperHessenberg )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j+2 : m); ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::UpperTriangle )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j+1 : m); ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::StrictUpper )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::LowerHessenberg )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j-1 : 0); i < m; ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::LowerTriangle )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::StrictLower )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j+1; i < m; ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
    else // if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::Dense )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if( isinf( A(i,j) ) )
                    return true;
        return false;
    }
}

/**
 * Returns true if and only if A has an infinite entry.
 * 
 * Specific implementation for band access types.
 * @see hasinf( access_t accessType, const matrix_t& A ).
 */
template< class matrix_t >
bool hasinf( band_t accessType, const matrix_t& A ) {

    using idx_t  = size_type< matrix_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ((j >= ku) ? (j-ku) : 0); i < min(m,j+kl+1); ++i)
            if( isinf( A(i,j) ) )
                return true;
    return false;
}

/**
 * Returns true if and only if x has an infinite entry.
 * 
 * @param[in] x vector.
 * 
 * @return true if x has an infinite entry.
 * @return false if x has no infinite entry.
 */
template< class vector_t >
bool hasinf( const vector_t& x ) {

    using idx_t  = size_type< vector_t >;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        if( isinf( x[i] ) )
            return true;
    return false;
}

// -----------------------------------------------------------------------------
/// isnan for complex numbers
template< typename real_t >
inline bool isnan( const std::complex<real_t>& x )
{
    return isnan( real(x) ) || isnan( imag(x) );
}

/**
 * Returns true if and only if A has an NaN entry.
 * 
 * @tparam access_t Type of access inside the algorithm.
 *      Either MatrixAccessPolicy or any type that implements
 *          operator MatrixAccessPolicy().
 * 
 * @param[in] accessType Determines the entries of A that will be checked.
 *      The following access types are allowed:
 *          MatrixAccessPolicy::Dense,
 *          MatrixAccessPolicy::UpperHessenberg,
 *          MatrixAccessPolicy::LowerHessenberg,
 *          MatrixAccessPolicy::UpperTriangle,
 *          MatrixAccessPolicy::LowerTriangle,
 *          MatrixAccessPolicy::StrictUpper,
 *          MatrixAccessPolicy::StrictLower.
 * 
 * @param[in] A matrix.
 * 
 * @return true if A has an NaN entry.
 * @return false if A has no NaN entry.
 */
template< class access_t, class matrix_t >
bool hasnan( access_t accessType, const matrix_t& A ) {

    using idx_t  = size_type< matrix_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::UpperHessenberg )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j+2 : m); ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::UpperTriangle )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j+1 : m); ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::StrictUpper )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::LowerHessenberg )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j > 1) ? j-1 : 0); i < m; ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::LowerTriangle )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j; i < m; ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::StrictLower )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j+1; i < m; ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
    else // if ( (MatrixAccessPolicy) accessType == MatrixAccessPolicy::Dense )
    {
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                if( isnan( A(i,j) ) )
                    return true;
        return false;
    }
}

/**
 * Returns true if and only if A has an NaN entry.
 * 
 * Specific implementation for band access types.
 * @see hasnan( access_t accessType, const matrix_t& A ).
 */
template< class matrix_t >
bool hasnan( band_t accessType, const matrix_t& A ) {

    using idx_t  = size_type< matrix_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = ((j >= ku) ? (j-ku) : 0); i < min(m,j+kl+1); ++i)
            if( isnan( A(i,j) ) )
                return true;
    return false;
}

/**
 * Returns true if and only if x has an NaN entry.
 * 
 * @param[in] x vector.
 * 
 * @return true if x has an NaN entry.
 * @return false if x has no NaN entry.
 */
template< class vector_t >
bool hasnan( const vector_t& x ) {

    using idx_t  = size_type< vector_t >;

    // constants
    const idx_t n = size(x);

    for (idx_t i = 0; i < n; ++i)
        if( isnan( x[i] ) )
            return true;
    return false;
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

template< class C1, class C2, class... Cs >
constexpr bool has_compatible_layout = 
    has_compatible_layout<C1, C2> &&
    has_compatible_layout<C1, Cs...> &&
    has_compatible_layout<C2, Cs...>;

template< class C1, class C2 >
constexpr bool has_compatible_layout< C1, C2 > = (
    ( layout<C1> == Layout::Unspecified ) ||
    ( layout<C2> == Layout::Unspecified ) ||
    ( layout<C1> == layout<C2> )
);

template< class C1, class T1, class C2, class T2 >
constexpr bool has_compatible_layout< pair<C1,T1>, pair<C2,T2> > =
    has_compatible_layout< C1, C2 >;

namespace internal {

    /**
     * @brief Trait to determine if a given list of data allows optimization
     * using a optimized BLAS library.
     */
    template<class...>
    struct allow_optblas {
        static constexpr bool value = false;    ///< True if the list of types
                                                ///< allows optimized BLAS library.
    };

    /**
     * @brief Auxiliary for matrices and vectors.
     */
    template<class, class = int>
    struct allow_optblas_aux {
        static constexpr bool value = false;
    };

    template<class C>
    struct allow_optblas<C> {
        static constexpr bool value             ///< True if the type
            = allow_optblas_aux<C,int>::value;  ///< allows optimized BLAS library.
    };
}

/// Alias for @c allow_optblas<>::value.
template<class... Ts>
constexpr bool allow_optblas_v = internal::allow_optblas< Ts... >::value;

namespace internal {

    template< class matrix_t >
    struct allow_optblas_aux< matrix_t,
        enable_if_t<
            is_matrix< matrix_t > &&
            !is_same_v<
                decltype( legacy_matrix( std::declval<matrix_t>() ) )
            , void >
        , int >
    > {
        static constexpr bool value =
            allow_optblas_v< type_t<matrix_t> > &&
            (
                ( layout<matrix_t> == Layout::ColMajor ) ||
                ( layout<matrix_t> == Layout::RowMajor )
            );
    };

    template< class vector_t >
    struct allow_optblas_aux< vector_t,
        enable_if_t<
            is_vector< vector_t > &&
            !is_same_v<
                decltype( legacy_vector( std::declval<vector_t>() ) )
            , void >
        , int >
    > {
        static constexpr bool value = allow_optblas_v< type_t<vector_t> >;
    };

    template< class C, class T >
    struct allow_optblas< pair<C,T> > {
        static constexpr bool value = 
            allow_optblas_v<T>
            && (
                (is_matrix<C> || is_vector<C>)
                ? (
                    allow_optblas_v<C> &&
                    is_same_v< type_t<C>, typename std::decay<T>::type >
                )
                : std::is_convertible< C, T >::value
            );
    };

    template< class C1, class T1, class C2, class T2, class... Ps >
    struct allow_optblas< pair<C1,T1>, pair<C2,T2>, Ps... > {
        static constexpr bool value = 
            allow_optblas_v< pair<C1,T1> > &&
            allow_optblas_v< pair<C2,T2>, Ps... > &&
            has_compatible_layout< C1, C2, Ps... >;
    };
}

template<class T1, class... Ts>
using enable_if_allow_optblas_t = enable_if_t<(
    allow_optblas_v< T1, Ts... >
), int >;

template<class T1, class... Ts>
using disable_if_allow_optblas_t = enable_if_t<(
    ! allow_optblas_v< T1, Ts... >
), int >;

#define TLAPACK_OPT_TYPE( T ) \
    namespace internal { \
        template<> struct allow_optblas< T > { \
            static constexpr bool value = true; \
        }; \
        template<> struct type_trait< T > { \
            using type = T; \
        }; \
    }

    /// Optimized types
    #ifdef USE_LAPACKPP_WRAPPERS
        TLAPACK_OPT_TYPE(float)
        TLAPACK_OPT_TYPE(double)
        TLAPACK_OPT_TYPE(std::complex<float>)
        TLAPACK_OPT_TYPE(std::complex<double>)
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

// -----------------------------------------------------------------------------
// Options:

/**
 * @brief Allocates workspace if needed and updates the workspace size
 * 
 * @param[out] work
 *      If `opts_lwork <= 0`, lwork bytes are allocated in work.
 *      If `opts_lwork > 0`, work is kept unchanged.
 * @param[in,out] lwork
 *      On the input: Workspace size needed.
 *      On the output: Available workspace size.
 * @param[in] opts_work     Optional workspace previously allocated.
 * @param[in] opts_lwork    Size of the optional workspace previously allocated.
 * 
 * @return byte*
 *      Returns a pointer to the workspace in case of a success.
 *      Returns a null pointer if `opts_lwork < lwork`.
 */
inline Workspace
alloc_workspace( vectorOfBytes& v, size_t lwork )
{
    v = vectorOfBytes( lwork ); // Allocates space in memory
    return Workspace( v.data(), v.size() );
}
inline Workspace
alloc_workspace( vectorOfBytes& v, size_t lwork, const Workspace& opts_w )
{
    if( opts_w.size() <= 0 )
    {
        return alloc_workspace( v, lwork );
    }
    else if( opts_w.size() < lwork )
    {
        tlapack_error( -4,
            std::string("Insuficient workspace.") +
            " Required: " + std::to_string(lwork)
            + ". Provided: " + std::to_string(opts_w.size())
        );
        return Workspace();
    }
    else
    {
        return Workspace( opts_w );
    }
}

template< class T, class idx_t, Layout L >
inline constexpr legacyMatrix<byte>
matrix_in_bytes( const legacyMatrix<T,idx_t,L>& A ) {
    if( L == Layout::ColMajor )
        return legacyMatrix<byte>( A.m * sizeof(T), A.n, (byte*) A.ptr, A.ldim * sizeof(T) );
    else // if( L == Layout::RowMajor )
        return legacyMatrix<byte>( A.n * sizeof(T), A.m, (byte*) A.ptr, A.ldim * sizeof(T) );
}
inline constexpr legacyMatrix<byte>
matrix_in_bytes( const legacyMatrix<byte>& A ) noexcept {
    return A;
}

/// Used to inform Undefined Workspace type
class undefined_t {};

/// Workspace options
template< class work_type = undefined_t >
struct workspace_opts_t
{
    using work_t = work_type;   ///< Workspace matrix type
    Workspace work;             ///< Workspace object

    inline constexpr
    workspace_opts_t( Workspace&& w = {} ) : work(w) { }

    // inline constexpr
    // workspace_opts_t( const Workspace& w ) : work(w) { }

    inline constexpr
    workspace_opts_t( const work_t& W )
    : work( matrix_in_bytes(legacy_matrix(W)) ) { }
};

template< class work_type, class work_default >
struct deduce_work {
    using type = work_type;
};

template< class work_default >
struct deduce_work<undefined_t,work_default> {
    using type = work_default;
};

template< class work_type, class work_default >
using deduce_work_t = typename deduce_work<work_type,work_default>::type;

template< class work_t >
inline constexpr
workspace_opts_t<work_t>
create_workspace_opts( const work_t& W ){ return workspace_opts_t<work_t>(W); }

} // namespace tlapack

#endif // TLAPACK_UTILS_HH
