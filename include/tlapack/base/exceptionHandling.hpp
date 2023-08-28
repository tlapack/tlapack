/// @file exceptionHandling.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_EXCEPTION_HH
#define TLAPACK_EXCEPTION_HH

#include <iostream>
#include <stdexcept>
#include <string>

/** @def TLAPACK_DEFAULT_INFCHECK
 * @brief Default behavior of checks for Infs. Checks can be
 * activated/deactivated at runtime
 *
 * @note Only used if TLAPACK_ENABLE_INFCHECK is defined and TLAPACK_NDEBUG
 * is not defined.
 *
 * @ingroup exception
 */
#ifndef TLAPACK_DEFAULT_INFCHECK
    #define TLAPACK_DEFAULT_INFCHECK true
#endif

/** @def TLAPACK_DEFAULT_NANCHECK
 * @brief Default behavior of checks for NaNs. Checks can be
 * activated/deactivated at runtime
 *
 * @note Only used if TLAPACK_ENABLE_NANCHECK is defined and TLAPACK_NDEBUG
 * is not defined.
 *
 * @ingroup exception
 *
 */
#ifndef TLAPACK_DEFAULT_NANCHECK
    #define TLAPACK_DEFAULT_NANCHECK true
#endif

namespace tlapack {
namespace internal {

    /**
     * @brief Create a string with the error message.
     *
     * @param info Error code.
     * @param detailedInfo Details about the error.
     * @return std::string Final error message.
     *
     * @ingroup exception
     */
    inline std::string error_msg(int info, const std::string& detailedInfo)
    {
        return std::string("[") + std::to_string(info) + "] " + detailedInfo;
    }
}  // namespace internal

/**
 * @brief Descriptor for Exception Handling
 * @ingroup exception
 */
struct ErrorCheck {
    bool inf = TLAPACK_DEFAULT_INFCHECK;  ///< Default behavior of inf check
    bool nan = TLAPACK_DEFAULT_NANCHECK;  ///< Default behavior of nan check
    bool internal = true;  ///< Used to enable / disable internal checks.
};

/**
 * @brief Options to disable error checking.
 * @ingroup exception
 */
constexpr ErrorCheck NO_ERROR_CHECK = {false, false, false};

/**
 * @brief Options for error checking.
 * @ingroup exception
 */
struct EcOpts {
    ErrorCheck ec = {};

    constexpr EcOpts(const ErrorCheck& ec_ = {}) : ec(ec_) {}
};
}  // namespace tlapack

// -----------------------------------------------------------------------------
// Macros to handle error in the input

#if defined(TLAPACK_CHECK_INPUT) && !defined(TLAPACK_NDEBUG)

    /**
     * @brief Throw an error if cond is false.
     *
     * ex: lapack_check( 1 > 2 ); throws an error.
     *
     * @note Disable the check by defining TLAPACK_NDEBUG or
     *  undefining TLAPACK_CHECK_INPUT.
     *
     * @ingroup exception
     */
    #define tlapack_check(cond)                                           \
        do {                                                              \
            if (!static_cast<bool>(cond)) throw std::domain_error(#cond); \
        } while (false)

    /**
     * @brief Throw an error if cond is true.
     *
     * ex: tlapack_check_false( 1 < 2 ); throws an error.
     *
     * @note Disable the check by defining TLAPACK_NDEBUG or
     *  undefining TLAPACK_CHECK_INPUT.
     *
     * @ingroup exception
     */
    #define tlapack_check_false(cond)                                    \
        do {                                                             \
            if (static_cast<bool>(cond)) throw std::domain_error(#cond); \
        } while (false)

#else  // !defined(TLAPACK_CHECK_INPUT) || defined(TLAPACK_NDEBUG)

    // <T>LAPACK does not check input parameters

    #define tlapack_check_false(cond) ((void)0)
    #define tlapack_check(cond) ((void)0)

#endif

// -----------------------------------------------------------------------------
// Macros to handle internal errors and warnings

#ifndef TLAPACK_NDEBUG

    /**
     * @brief Error handler
     *
     * @param[in] info Code of the error.
     * @param[in] detailedInfo String with information about the error.
     *
     * @note Disable the handler by defining TLAPACK_NDEBUG.
     *
     * @ingroup exception
     */
    #define tlapack_error(info, detailedInfo) \
        throw std::runtime_error(             \
            tlapack::internal::error_msg(info, detailedInfo))

    /**
     * @brief Warning handler
     *
     * @param[in] info Code of the warning.
     * @param[in] detailedInfo String with information about the warning.
     *
     * @note Disable the handler by defining TLAPACK_NDEBUG.
     *
     * @ingroup exception
     */
    #define tlapack_warning(info, detailedInfo)                       \
        std::cerr << tlapack::internal::error_msg(info, detailedInfo) \
                  << std::endl;

    /**
     * @brief Error handler with conditional
     *
     * @param[in] cond If true, tlapack error is thrown.
     * @param[in] info Code of the error.
     * @param[in] detailedInfo String with information about the error.
     *
     * @note Disable the handler by defining TLAPACK_NDEBUG.
     *
     * @ingroup exception
     */
    #define tlapack_error_if(cond, info, detailedInfo)                      \
        do {                                                                \
            if (static_cast<bool>(cond)) tlapack_error(info, detailedInfo); \
        } while (false)

#else

    // <T>LAPACK does not throw errors or display warnings

    #define tlapack_error(info, detailedInfo) ((void)0)
    #define tlapack_warning(info, detailedInfo) ((void)0)
    #define tlapack_error_if(cond, info, detailedInfo) ((void)0)

#endif

#endif  // TLAPACK_EXCEPTION_HH
