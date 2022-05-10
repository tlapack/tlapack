// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_EXCEPTION_HH__
#define __TLAPACK_EXCEPTION_HH__

#include <stdexcept>
#include <string>
#include <iostream>
#include <cassert>

#ifndef TLAPACK_DEFAULT_INFCHECK
    #define TLAPACK_DEFAULT_INFCHECK 0
#endif

#ifndef TLAPACK_DEFAULT_NANCHECK
    #define TLAPACK_DEFAULT_NANCHECK 0
#endif

namespace tlapack {

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
        bool root = true; ///< Used to enable / disable some checks on recursive calls.
        
        // /** Search for infs and nans in the input and ouput of internal calls.
        //  * 
        //  * - 0 : No internal check.
        //  * - k : Internal check on k levels in the call tree.
        //  */
        // std::size_t infnanInternalCheck = std::numeric_limits<std::size_t>::max();

        inline ErrorCheck leaf() const {
            ErrorCheck ec = *this;
            ec.root = false;
            return ec;
        }
    };

}

// -----------------------------------------------------------------------------
// Macros to handle error in the input

#if defined(TLAPACK_CHECK_INPUT) && !defined(TLAPACK_NDEBUG)

    /// ex: lapack_check( 1 < 2, -6 );
    #define tlapack_check( cond ) do { \
        if( !static_cast<bool>(cond) ) \
            throw std::domain_error( #cond ); \
    } while(false)

    /// ex: tlapack_check_false( 2 < 1, -6 );
    #define tlapack_check_false( cond, ... ) do { \
        if( static_cast<bool>(cond) ) \
            throw std::domain_error( #cond ); \
    } while(false)

#else // !defined(TLAPACK_CHECK_INPUT) || defined(TLAPACK_NDEBUG)

    // <T>LAPACK does not check input parameters

    #define tlapack_check_false( cond, ... ) \
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
        throw std::runtime_error( \
            tlapack::internal::error_msg(info, detailedInfo) )

    /**
     * @brief Error handler with conditional
     * 
     * @param[in] check If true, throw an exception.
     * @param[in] info Code of the error.
     * @param[in] detailedInfo String with information about the error.
     */
    #define tlapack_error_if( check, info, detailedInfo ) do { \
        if( static_cast<bool>(check) ) \
            tlapack_error( info, detailedInfo ); \
    } while(false)

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

#else
    #define tlapack_error( check, info, detailedInfo ) \
        ((void)0)
    #define tlapack_warning( check, info, detailedInfo ) \
        ((void)0)
#endif

// -----------------------------------------------------------------------------
// Macros to handle Infs warnings

#if defined(TLAPACK_ENABLE_INFCHECK) && !defined(TLAPACK_NDEBUG)
    #define tlapack_warn_infs_in_matrix( check, accessType, A, info, detailedInfo ) do { \
        if( static_cast<bool>(check) && hasinf(accessType, A) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)

    #define tlapack_warn_infs_in_vector( check, x, info, detailedInfo ) do { \
        if( static_cast<bool>(check) && hasinf(x) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)
#else // !defined(TLAPACK_ENABLE_INFCHECK) || defined(TLAPACK_NDEBUG)
    #define tlapack_warn_infs_in_matrix( check, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_warn_infs_in_vector( check, x, info, detailedInfo ) \
        ((void)0)
#endif

// -----------------------------------------------------------------------------
// Macros to handle NaNs warnings

#if defined(TLAPACK_ENABLE_NANCHECK) && !defined(TLAPACK_NDEBUG)
    #define tlapack_warn_nans_in_matrix( check, accessType, A, info, detailedInfo ) do { \
        if( static_cast<bool>(check) && hasnan(accessType, A) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)

    #define tlapack_warn_nans_in_vector( check, x, info, detailedInfo ) do { \
        if( static_cast<bool>(check) && hasnan(x) ) \
            tlapack_warning( info, detailedInfo ); \
    } while(false)
#else // !defined(TLAPACK_ENABLE_NANCHECK) || defined(TLAPACK_NDEBUG)
    #define tlapack_warn_nans_in_matrix( check, accessType, A, info, detailedInfo ) \
        ((void)0)
    #define tlapack_warn_nans_in_vector( check, x, info, detailedInfo ) \
        ((void)0)
#endif

#endif // __TLAPACK_EXCEPTION_HH__