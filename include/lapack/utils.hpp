// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

#include <utility>
#include <type_traits>
#include <stdexcept>
#include "blas/arrayTraits.hpp"

// -----------------------------------------------------------------------------
// Macros to handle error checks

#if defined(LAPACK_ERROR_NDEBUG) || defined(NDEBUG)

    // <T>BLAS does no error checking;
    // lower level BLAS may still handle errors via xerbla

    #define lapack_error_if( cond, info ) \
        ((void)0)
    #define tlapack_check( check, cond, info ) \
        ((void)0)

#else

    /// ex: lapack_error_if( a < b, -6 );
    #define lapack_error_if( cond, info ) do { \
        if( static_cast<bool>(cond) ) \
            throw std::domain_error( "[" #info "] " #cond ); \
    } while(false)

    #define tlapack_check( check, cond, info ) do { \
        if( static_cast<bool>(check) && !static_cast<bool>(cond) ) \
            throw std::domain_error( "[" #info "] " #cond ); \
    } while(false)

#endif

#ifdef TLAPACK_CHECK_PARAM
    #define tlapack_check_param( cond, info ) tlapack_check( opts.paramCheck, cond, info )
#else
    #define tlapack_check_param( cond, info ) ((void)0)
#endif

#ifdef TLAPACK_CHECK_ACCESS
    #define tlapack_check_access( cond, info ) tlapack_check( opts.accessCheck, cond, info )
#else
    #define tlapack_check_access( cond, info ) ((void)0)
#endif

#ifdef TLAPACK_CHECK_SIZES
    #define tlapack_check_sizes( cond, info ) tlapack_check( opts.sizeCheck, cond, info )
#else
    #define tlapack_check_sizes( cond, info ) ((void)0)
#endif

namespace lapack {
    template< class detailedInfo_t >
    void report( int info, const detailedInfo_t& detailedInfo ) { }

    template< class access_t, class matrix_t >
    bool hasinf( access_t accessType, const matrix_t& A ) {

        using idx_t  = size_type< matrix_t >;
        using blas::isinf;

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

    template< class matrix_t >
    bool hasinf( band_t accessType, const matrix_t& A ) {

        using idx_t  = size_type< matrix_t >;
        using blas::isinf;

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

    template< class vector_t >
    bool hasinf( const vector_t& x ) {

        using idx_t  = size_type< vector_t >;
        using blas::isinf;

        // constants
        const idx_t n = size(x);

        for (idx_t i = 0; i < n; ++i)
            if( isinf( x[i] ) )
                return true;
        return false;
    }

    template< class access_t, class matrix_t >
    bool hasnan( access_t accessType, const matrix_t& A ) {

        using idx_t  = size_type< matrix_t >;
        using blas::isnan;

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

    template< class matrix_t >
    bool hasnan( band_t accessType, const matrix_t& A ) {

        using idx_t  = size_type< matrix_t >;
        using blas::isnan;

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

    template< class vector_t >
    bool hasnan( const vector_t& x ) {

        using idx_t  = size_type< vector_t >;
        using blas::isnan;

        // constants
        const idx_t n = size(x);

        for (idx_t i = 0; i < n; ++i)
            if( isnan( x[i] ) )
                return true;
        return false;
    }
}

#if defined(LAPACK_ERROR_NDEBUG)
    #define tlapack_report( info, detailedInfo ) \
        ((void)0)
#else
    #define tlapack_report( info, detailedInfo ) \
        lapack::report( info, detailedInfo )
#endif

#ifdef TLAPACK_CHECK_INFS
    #define tlapack_report_infs_in_matrix( check, accessType, A, info, detailedInfo ) do { \
        if( check && hasinf(accessType, A) ) \
            tlapack_report( info, detailedInfo ); \
    } while(false)

    #define tlapack_report_infs_in_vector( check, x, info, detailedInfo ) do { \
        if( check && hasinf(x) ) \
            tlapack_report( info, detailedInfo ); \
    } while(false)
#else
    #define tlapack_report_infs_in_matrix( check, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_report_infs_in_vector( check, x, info, detailedInfo ) \
        ((void)0)
#endif

#ifdef TLAPACK_CHECK_NANS
    #define tlapack_report_nans_in_matrix( check, accessType, A, info, detailedInfo ) do { \
        if( check && hasnan(accessType, A) ) \
            tlapack_report( info, detailedInfo ); \
    } while(false)

    #define tlapack_report_nans_in_vector( check, x, info, detailedInfo ) do { \
        if( check && hasnan(x) ) \
            tlapack_report( info, detailedInfo ); \
    } while(false)
#else
    #define tlapack_report_nans_in_matrix( check, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_report_nans_in_vector( check, x, info, detailedInfo ) \
        ((void)0)
#endif

namespace lapack {

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

using blas::access_granted;
using blas::access_denied;
using blas::exception_t;

} // namespace lapack

#endif // __TLAPACK_UTILS_HH__
