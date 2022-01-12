// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_UTILS_HH__
#define __TLAPACK_UTILS_HH__

namespace blas {
    // Forward declaration
    void error( const char* error_msg, const char* func );
}

namespace lapack {
    using blas::size;
    using blas::nrows;
    using blas::ncols;
    using blas::submatrix;
    using blas::subvector;
    using blas::row;
    using blas::col;
    using blas::diag;
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

#endif // __TLAPACK_UTILS_HH__