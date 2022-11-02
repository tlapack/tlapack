// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EXCEPTION_HH
#define TLAPACK_EXCEPTION_HH

#include <stdexcept>
#include <string>
#include <iostream>

#ifndef TLAPACK_DEFAULT_INFCHECK
    #define TLAPACK_DEFAULT_INFCHECK true
#endif

#ifndef TLAPACK_DEFAULT_NANCHECK
    #define TLAPACK_DEFAULT_NANCHECK true
#endif

namespace tlapack {

    using check_error = std::domain_error;
    using internal_error = std::runtime_error;

    namespace internal {
        inline
        std::string error_msg( int info, const std::string& detailedInfo ) {
            return std::string("[") + std::to_string(info) + "] " + detailedInfo;
        }
    }

    /// Descriptor for Exception Handling
    struct ErrorCheck {
        
        bool inf  = TLAPACK_DEFAULT_INFCHECK; ///< Default behavior of inf check in the routines of <T>LAPACK.
        bool nan  = TLAPACK_DEFAULT_NANCHECK; ///< Default behavior of nan check in the routines of <T>LAPACK.
        bool internal = true; ///< Used to enable / disable internal checks.

    };

    /// It has all errors turned off
    constexpr ErrorCheck noErrorCheck = { false, false, false };

    /// Error Checking options
    struct ec_opts_t {
        ErrorCheck ec = {};
        
        inline constexpr
        ec_opts_t( const ErrorCheck& ec_ = {} ) : ec(ec_) {}
    };
}

// -----------------------------------------------------------------------------
// Macros to handle error in the input

#if defined(TLAPACK_CHECK_INPUT) && !defined(TLAPACK_NDEBUG)

    /**
     * @brief Throw an error if cond is false.
     * 
     * ex: lapack_check( 1 > 2 ); throws an error.
     */
    #define tlapack_check( cond ) do { \
        if( !static_cast<bool>(cond) ) \
            throw tlapack::check_error( #cond ); \
    } while(false)

    /**
     * @brief Throw an error if cond is true.
     * 
     * ex: tlapack_check_false( 1 < 2 ); throws an error.
     */
    #define tlapack_check_false( cond ) do { \
        if( static_cast<bool>(cond) ) \
            throw tlapack::check_error( #cond ); \
    } while(false)

#else // !defined(TLAPACK_CHECK_INPUT) || defined(TLAPACK_NDEBUG)

    // <T>LAPACK does not check input parameters

    #define tlapack_check_false( cond ) \
        ((void)0)
    #define tlapack_check( cond ) \
        ((void)0)

#endif

// -----------------------------------------------------------------------------
// Macros to handle internal errors and warnings

#ifndef TLAPACK_NDEBUG

    /**
     * @brief Error handler
     * 
     * @param[in] info Code of the error.
     * @param[in] detailedInfo String with information about the error.
     */
    #define tlapack_error( info, detailedInfo ) \
        throw tlapack::internal_error( \
            tlapack::internal::error_msg(info, detailedInfo) )

    /**
     * @brief Warning handler
     * 
     * @param[in] info Code of the warning.
     * @param[in] detailedInfo String with information about the warning.
     */
    #define tlapack_warning( info, detailedInfo ) \
        std::cerr \
            << tlapack::internal::error_msg(info, detailedInfo) \
            << std::endl;

    /**
     * @brief Error handler with conditional for internal checks
     * 
     * @param[in] ec Exception handling configuration at runtime.
     *      Default options are defined in ErrorCheck.
     * @param[in] info Code of the error.
     * @param[in] detailedInfo String with information about the error.
     */
    #define tlapack_error_internal( ec, info, detailedInfo ) do { \
        if( static_cast<bool>(ec.internal) ) \
            tlapack_error( info, detailedInfo ); \
    } while(false)

#else

    // <T>LAPACK does not throw errors or display warnings

    #define tlapack_error( info, detailedInfo ) \
        ((void)0)
    #define tlapack_warning( info, detailedInfo ) \
        ((void)0)
    #define tlapack_error_internal( ec, info, detailedInfo ) \
        ((void)0)
#endif

// -----------------------------------------------------------------------------
// Macros to handle Infs warnings

#if defined(TLAPACK_ENABLE_INFCHECK) && !defined(TLAPACK_NDEBUG)

    #define tlapack_warn_infs_in_matrix( ec, accessType, A, info, detailedInfo ) do { \
        if( static_cast<bool>(ec.inf) && hasinf(accessType, A) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)

    #define tlapack_warn_infs_in_vector( ec, x, info, detailedInfo ) do { \
        if( static_cast<bool>(ec.inf) && hasinf(x) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)

#else // !defined(TLAPACK_ENABLE_INFCHECK) || defined(TLAPACK_NDEBUG)

    // <T>LAPACK does not check for infs

    #define tlapack_warn_infs_in_matrix( ec, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_warn_infs_in_vector( ec, x, info, detailedInfo ) \
        ((void)0)
#endif

// -----------------------------------------------------------------------------
// Macros to handle NaNs warnings

#if defined(TLAPACK_ENABLE_NANCHECK) && !defined(TLAPACK_NDEBUG)

    #define tlapack_warn_nans_in_matrix( ec, accessType, A, info, detailedInfo ) do { \
        if( static_cast<bool>(ec.nan) && hasnan(accessType, A) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)

    #define tlapack_warn_nans_in_vector( ec, x, info, detailedInfo ) do { \
        if( static_cast<bool>(ec.nan) && hasnan(x) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)
    
#else // !defined(TLAPACK_ENABLE_NANCHECK) || defined(TLAPACK_NDEBUG)

    // <T>LAPACK does not check for nans

    #define tlapack_warn_nans_in_matrix( ec, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_warn_nans_in_vector( ec, x, info, detailedInfo ) \
        ((void)0)
#endif

#endif // TLAPACK_EXCEPTION_HH
