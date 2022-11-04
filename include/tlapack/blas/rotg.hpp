// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_ROTG_HH
#define TLAPACK_BLAS_ROTG_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Construct plane rotation that eliminates b, such that:
 * \[
 *       \begin{bmatrix} r     \\ 0      \end{bmatrix}
 *     := \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} a     \\ b      \end{bmatrix}.
 * \]
 *
 * @see rot to apply the rotation.
 *
 * @param[in,out] a On entry, scalar a. On exit, set to r.
 * @param[in,out] b On entry, scalar b. On exit, set to s, 1/c, or 0.
 * @param[out]    c Cosine of rotation; real.
 * @param[in]     s Sine of rotation;   real.
 *
 * @ingroup rotg
 */
template <typename T,
    enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
    disable_if_allow_optblas_t< T > = 0
>
void rotg(
    T& a, T& b,
    T& c, T& s )
{
    // Constants
    const T one  = 1;
    const T zero = 0;

    // Scaling constants
    const T safmin = safe_min<T>();
    const T safmax = safe_max<T>();

    // Norms
    const T anorm = tlapack::abs(a);
    const T bnorm = tlapack::abs(b);

    // quick return
    if ( bnorm == zero ) {
        c = one;
        s = zero;
        b = zero;
    }
    else if ( anorm == zero ) {
        c = zero;
        s = one;
        a = b;
        b = one;
    }
    else {
        T scl = min( safmax, max(safmin, anorm, bnorm) );
        T sigma = (anorm > bnorm)
            ? sgn(a)
            : sgn(b);
        T r = sigma * scl * sqrt( (a/scl) * (a/scl) + (b/scl) * (b/scl) );
        c = a / r;
        s = b / r;
        a = r;
        if ( anorm > bnorm )
            b = s;
        else if ( c != zero )
            b = one / c;
        else
            b = one;
    }
}

/**
 * Construct plane rotation that eliminates b, such that:
 * \[
 *       \begin{bmatrix} r     \\ 0      \end{bmatrix}
 *     = \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
 *       \begin{bmatrix} a     \\ b      \end{bmatrix}.
 * \]
 *
 * @see rot to apply the rotation.
 *
 * @param[in,out] a On entry, scalar a. On exit, set to r.
 * @param[in]     b Scalar b.
 * @param[out]    c Cosine of rotation; real.
 * @param[in]     s Sine of rotation; complex.
 *
 * @details
 *
 * Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
 * ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
 *
 * @ingroup rotg
 */
template <typename T,
    enable_if_t< !is_same_v< T, real_type<T> >, int > = 0,
    disable_if_allow_optblas_t< T > = 0
>
void rotg(
    T& a, const T& b,
    real_type<T>& c,
    T& s )
{
    typedef real_type<T> real_t;
    typedef T scalar_t;

    // Constants
    const real_t r_one = 1;
    const real_t r_zero = 0;
    const scalar_t zero = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();
    const real_t rtmin  = root_min<real_t>();
    const real_t rtmax  = root_max<real_t>();

    // quick return
    if ( b == zero ) {
        c = r_one;
        s = zero;
        return;
    }

    if ( a == zero ) {
        c = r_zero;
        real_t g1 = max( tlapack::abs(real(b)), tlapack::abs(imag(b)) );
        if ( g1 > rtmin && g1 < rtmax ) {
            // Use unscaled algorithm
            real_t g2 = real(b)*real(b) + imag(b)*imag(b);
            real_t d = sqrt( g2 );
            s = conj( b ) / d;
            a = d;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = b*uu;
            real_t g2 = real(gs)*real(gs) + imag(gs)*imag(gs);
            real_t d = sqrt( g2 );
            s = conj( gs ) / d;
            a = d*u;
        }
    }
    else {
        real_t f1 = max( tlapack::abs(real(a)), tlapack::abs(imag(a)) );
        real_t g1 = max( tlapack::abs(real(b)), tlapack::abs(imag(b)) );
        if ( f1 > rtmin && f1 < rtmax &&
            g1 > rtmin && g1 < rtmax ) {
            // Use unscaled algorithm
            real_t f2 = real(a)*real(a) + imag(a)*imag(a);
            real_t g2 = real(b)*real(b) + imag(b)*imag(b);
            real_t h2 = f2 + g2;
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            c  = f2*p;
            s  = conj( b )*( a*p );
            a *= h2*p ;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, f1, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = b*uu;
            real_t g2 = real(gs)*real(gs) + imag(gs)*imag(gs);
            real_t f2, h2, w;
            scalar_t fs;
            if ( f1*uu < rtmin ) {
                // a is not well-scaled when scaled by g1.
                real_t v = min( safmax, max( safmin, f1 ) );
                real_t vv = r_one / v;
                w = v * uu;
                fs = a*vv;
                f2 = real(fs)*real(fs) + imag(fs)*imag(fs);
                h2 = f2*w*w + g2;
            }
            else {
                // Otherwise use the same scaling for a and b.
                w = r_one;
                fs = a*uu;
                f2 = real(fs)*real(fs) + imag(fs)*imag(fs);
                h2 = f2 + g2;
            }
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            c = ( f2*p )*w;
            s = conj( gs )*( fs*p );
            a = ( fs*( h2*p ) )*u;
        }
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

    template <typename T,
        enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
        enable_if_allow_optblas_t< T > = 0
    >
    inline
    void rotg( T& a, T& b, T& c, T& s )
    {
        // Constants
        const T zero = 0;
        const T one  = 1;
        const T anorm = tlapack::abs(a);
        const T bnorm = tlapack::abs(b);

        T r;
        ::lapack::lartg( a, b, &c, &s, &r );
        
        // Return information on a and b:
        a = r;
        if( s == zero || c == zero || (anorm > bnorm) )
            b = s;
        else if ( c != zero )
            b = one / c;
        else
            b = one;
    }

    template <typename T,
        enable_if_t< !is_same_v< T, real_type<T> >, int > = 0,
        enable_if_allow_optblas_t< T > = 0
    >
    inline
    void rotg( T& a, const T& b, real_type<T>& c, T& s )
    {
        T r;
        ::lapack::lartg( a, b, &c, &s, &r );
        a = r;
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_ROTG_HH
