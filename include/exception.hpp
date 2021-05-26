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
        throw Error( condstr, func );
    }
}

#if defined(_MSC_VER)
    #define BLASPP_ATTR_FORMAT(I, F)
#else
    #define BLASPP_ATTR_FORMAT(I, F) __attribute__((format( printf, I, F )))
#endif

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
    BLASPP_ATTR_FORMAT(4, 5);

inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
inline void abort_if( bool cond, const char* func,  const char* format, ... )
    BLASPP_ATTR_FORMAT(3, 4);

inline void abort_if( bool cond, const char* func,  const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );

        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
    }
}

#undef BLASPP_ATTR_FORMAT

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

#elif defined(BLAS_ERROR_ASSERT)

    // blaspp aborts on error
    #define blas_error_if( cond ) \
        blas::internal::abort_if( cond, __func__, "%s", #cond )

    #define blas_error_if_msg( cond, ... ) \
        blas::internal::abort_if( cond, __func__, __VA_ARGS__ )

#else

    // blaspp throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // ex: blas_error_if( a < b );
    #define blas_error_if( cond ) \
        blas::internal::throw_if( cond, #cond, __func__ )

    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // ex: blas_error_if_msg( a < b, "a %d < b %d", a, b );
    #define blas_error_if_msg( cond, ... ) \
        blas::internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

} // namespace blas

#endif // __TLAPACK_EXCEPTION_HH__