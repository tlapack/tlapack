// Copyright (c) 2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_EXCEPTION_HH__
#define __TBLAS_EXCEPTION_HH__

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

    /// Constructs BLAS error with message: "msg, in function <func>"
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

// -----------------------------------------------------------------------------
/// Main function to handle errors in <T>BLAS
/// Default implementation: throw blas::Error( error_msg, func )
inline void error( const char* error_msg, const char* func ) {
    throw Error( error_msg, func );
}

// -----------------------------------------------------------------------------
// Internal helpers
namespace internal {

    // -------------------------------------------------------------------------
    /// internal helper function that calls blas::error if cond is true
    /// called by blas_error_if macro
    inline void error_if( bool cond, const char* condstr, const char* func )
    {
        if (cond)
            error( condstr, func );
    }

    // -------------------------------------------------------------------------
    /// internal helper function that calls blas::error if cond is true
    /// uses printf-style format for error message
    /// called by blas_error_if_msg macro
    /// condstr is ignored, but differentiates this from the other version.
    inline void error_if( bool cond, const char* condstr, const char* func,
        const char* format, ... )
    #ifndef _MSC_VER
        __attribute__((format( printf, 4, 5 )));
    #endif
    
    inline void error_if( bool cond, const char* condstr, const char* func,
        const char* format, ... )
    {
        if (cond) {
            char buf[80];
            va_list va;
            va_start( va, format );
            vsnprintf( buf, sizeof(buf), format, va );

            error( buf, func );
        }
    }

}  // namespace internal

} // namespace blas

// -----------------------------------------------------------------------------
// Macros to handle error checks
#if defined(BLAS_ERROR_NDEBUG) || defined(NDEBUG)

    // <T>BLAS does no error checking;
    // lower level BLAS may still handle errors via xerbla
    #define blas_error_if( cond ) \
        ((void)0)
    #define blas_error_if_msg( cond, ... ) \
        ((void)0)

#else

    /// internal macro to get strings: #cond and __func__
    /// ex: blas_error_if( a < b );
    #define blas_error_if( cond ) \
        blas::internal::error_if( cond, #cond, __func__ )

    /// internal macro takes cond and printf-style format for error message.
    /// ex: blas_error_if_msg( a < b, "a %d < b %d", a, b );
    #define blas_error_if_msg( cond, ... ) \
        blas::internal::error_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

#endif // __TBLAS_EXCEPTION_HH__