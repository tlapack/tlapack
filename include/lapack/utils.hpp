// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

#include <utility>
#include <type_traits>
#include "blas/arrayTraits.hpp"

namespace blas {
    // Forward declaration
    void error( const char* error_msg, const char* func );
}

// -----------------------------------------------------------------------------
// Macros to handle error checks
#if defined(LAPACK_ERROR_NDEBUG) || defined(NDEBUG)

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
        blas::error( msg, __func__ ); \
        return code; \
    } while(false)

    /// internal macro to get strings: #cond and __func__
    /// ex: lapack_error_if( a < b, -6 );
    /// @returns code if a < b
    #define lapack_error_if( cond, code ) do { \
        if( cond ) { \
            blas::error( #cond, __func__ ); \
            return code; \
        } \
    } while(false)

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

} // namespace lapack

#endif // __TLAPACK_UTILS_HH__