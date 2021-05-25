// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_EXCEPTION_HH__
#define __TLAPACK_EXCEPTION_HH__

#include <exception>
#include <string>
#include <cstdarg>

namespace blas {

// -----------------------------------------------------------------------------
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLAS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLAS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by blas_error_if macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
#ifndef BLAS_ERROR_ASSERT
        throw Error( condstr, func );
#else
        fprintf( stderr, "Error: %s, in function %s\n", condstr, func );
        abort();
#endif
    }
}

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
// condstr is ignored, but differentiates this from the other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
#ifndef _MSC_VER
    __attribute__((format( printf, 4, 5 )))
#endif
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );

        vsnprintf( buf, sizeof(buf), format, va );

#ifndef BLAS_ERROR_ASSERT
        throw Error( buf, func );
#else
        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
#endif
    }
}

}  // namespace internal

// -----------------------------------------------------------------------------
// internal macros to handle error checks
#if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))

    // blaspp does no error checking;
    // lower level BLAS may still handle errors via xerbla
    #define blas_error_if( cond ) \
        ((void)0)

    #define blas_error_if_msg( cond, ... ) \
        ((void)0)

#else

    // blaspp throws errors (default) or aborts (if BLAS_ERROR_ASSERT is defined)
    // internal macro to get string #cond; throws Error or aborts if cond is true
    // ex: blas_error_if( a < b );
    // (See https://www.math.utah.edu/docs/info/cpp_1.html#SEC23 to understand the `do {...} while (0);`) 
    #define blas_error_if( cond ) \
    do { if (cond) blas::internal::throw( cond, #cond, __func__ ); } \
    while (0)

    // internal macro takes cond and printf-style format for error message.
    // throws Error or aborts (when BLAS_ERROR_ASSERT is defined) if cond is true.
    // ex: blas_error_if_msg( a < b, "a %d < b %d", a, b );
    // (See https://www.math.utah.edu/docs/info/cpp_1.html#SEC23 to understand the `do {...} while (0);`) 
    #define blas_error_if_msg( cond, ... ) \
    do { if( cond ) blas::internal::throw( cond, #cond, __func__, __VA_ARGS__ ); } \
    while (0)

#endif

} // namespace blas

#endif // __TLAPACK_EXCEPTION_HH__