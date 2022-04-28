// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TBLAS_LEGACY_UTILS_HH__
#define __TBLAS_LEGACY_UTILS_HH__

#ifndef TLAPACK_USE_MDSPAN

    #include "legacy_api/blas/legacyArray.hpp"
    #include "plugins/tlapack_legacyArray.hpp" // Loads LegacyArray plugin

    #define tlapack_expr_with_2vectors( x, TX, n, X, incx, ... ) do { \
        using blas::internal::vector; \
        using blas::internal::backward_vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
        else if( incx == -1 ) { \
            auto x = backward_vector( (TX*) X, n ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
        else if( incx > 1 ) { \
            auto x = vector( (TX*) X, n, incx ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
        else { \
            auto x = backward_vector( (TX*) X, n, -incx ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
    } while(false)

    #define tlapack_expr_with_vector( x, TX, n, X, incx, expr ) do { \
        using blas::internal::vector; \
        using blas::internal::backward_vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            expr; \
        } \
        else if( incx == -1 ) { \
            auto x = backward_vector( (TX*) X, n ); \
            expr; \
        } \
        else if( incx > 1 ) { \
            auto x = vector( (TX*) X, n, incx ); \
            expr; \
        } \
        else { \
            auto x = backward_vector( (TX*) X, n, -incx ); \
            expr; \
        } \
    } while(false)

    #define tlapack_expr_with_vector_positiveInc( x, TX, n, X, incx, expr ) do { \
        using blas::internal::vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            expr; \
        } \
        else { \
            auto x = vector( (TX*) X, n, incx ); \
            expr; \
        } \
    } while(false)

#else
    #include "legacy_api/blas/mdspan.hpp"
    #include "plugins/tlapack_mdspan.hpp" // Loads mdspan plugin

    #define tlapack_expr_with_2vectors( x, TX, n, X, incx, ... ) do { \
        using blas::internal::vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
        else if( incx > 1 ) { \
            auto x = vector( (TX*) X, n, incx ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
        else { \
            auto x = vector( (TX*) &X[(1-n)*incx], n, incx ); \
            tlapack_expr_with_vector( __VA_ARGS__ ); \
        } \
    } while(false)

    #define tlapack_expr_with_vector( x, TX, n, X, incx, expr ) do { \
        using blas::internal::vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            expr; \
        } \
        else if( incx > 1 ) { \
            auto x = vector( (TX*) X, n, incx ); \
            expr; \
        } \
        else { \
            auto x = vector( (TX*) &X[(1-n)*incx], n, incx ); \
            expr; \
        } \
    } while(false)

    #define tlapack_expr_with_vector_positiveInc( x, TX, n, X, incx, expr ) do { \
        using blas::internal::vector; \
        if( incx == 1 ) { \
            auto x = vector( (TX*) X, n ); \
            expr; \
        } \
        else { \
            auto x = vector( (TX*) X, n, incx ); \
            expr; \
        } \
    } while(false)
#endif

#endif // __TBLAS_LEGACY_UTILS_HH__
