// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTG_HH
#define BLAS_ROTG_HH

#include "blas/utils.hpp"

namespace blas {

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
 * Generic implementation for arbitrary data types.
 *
 * @param[in, out] a
 *     On entry, scalar a. On exit, set to r.
 *
 * @param[in, out] b
 *     On entry, scalar b. On exit, set to s, 1/c, or 0.
 *
 * @param[out] c
 *     Cosine of rotation; real.
 *
 * @param[out] s
 *     Sine of rotation; real.
 *
 * __Further details__
 *
 * Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
 * ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
 *
 * @ingroup rotg
 */
template <typename real_t>
void rotg(
    real_t *a,
    real_t *b,
    real_t *c,
    real_t *s )
{
    #define A (*a)
    #define B (*b)
    #define C (*c)
    #define S (*s)

    // Constants
    const real_t one  = 1;
    const real_t zero = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();

    // Norms
    const real_t anorm = blas::abs(A);
    const real_t bnorm = blas::abs(B);

    // quick return
    if ( bnorm == zero ) {
        C = one;
        S = zero;
        B = zero;
    }
    else if ( anorm == zero ) {
        C = zero;
        S = one;
        A = B;
        B = one;
    }
    else {
        real_t scl = min( safmax, max(safmin, anorm, bnorm) );
        real_t sigma = (anorm > bnorm)
            ? sgn(A)
            : sgn(B);
        real_t r = sigma * scl * sqrt( (A/scl) * (A/scl) + (B/scl) * (B/scl) );
        C = A / r;
        S = B / r;
        A = r;
        if ( anorm > bnorm )
            B = S;
        else if ( C != zero )
            B = one / C;
        else
            B = one;
    }

    #undef A
    #undef B
    #undef C
    #undef S
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
 * Generic implementation for arbitrary data types.
 *
 * @param[in, out] a
 *     On entry, scalar a. On exit, set to r.
 *
 * @param[in, out] b
 *     On entry, scalar b. On exit, set to s, 1/c, or 0.
 *
 * @param[out] c
 *     Cosine of rotation; real.
 *
 * @param[out] s
 *     Sine of rotation; complex.
 *
 * __Further details__
 *
 * Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
 * ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
 *
 * @ingroup rotg
 */
template <typename T>
void rotg(
    T *a,
    T *b,
    blas::real_type<T>    *c,
    blas::complex_type<T> *s )
{
    typedef real_type<T> real_t;
    typedef complex_type<T> scalar_t;

    #define ABSSQ(t_) real(t_)*real(t_) + imag(t_)*imag(t_)
    #define A (*a)
    #define B (*b)
    #define C (*c)
    #define S (*s)

    // Constants
    const real_t r_one = 1;
    const real_t r_zero = 0;
    const scalar_t zero = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();
    const real_t rtmin = root_min<real_t>();
    const real_t rtmax = root_max<real_t>();

    // quick return
    if ( B == zero ) {
        C = r_one;
        S = zero;
        return;
    }

    if ( A == zero ) {
        C = r_zero;
        real_t g1 = max( blas::abs(real(B)), blas::abs(imag(B)) );
        if ( g1 > rtmin && g1 < rtmax ) {
            // Use unscaled algorithm
            real_t g2 = ABSSQ( B );
            real_t d = sqrt( g2 );
            S = conj( B ) / d;
            A = d;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = B*uu;
            real_t g2 = ABSSQ( gs );
            real_t d = sqrt( g2 );
            S = conj( gs ) / d;
            A = d*u;
        }
    }
    else {
        real_t f1 = max( blas::abs(real(A)), blas::abs(imag(A)) );
        real_t g1 = max( blas::abs(real(B)), blas::abs(imag(B)) );
        if ( f1 > rtmin && f1 < rtmax &&
            g1 > rtmin && g1 < rtmax ) {
            // Use unscaled algorithm
            real_t f2 = ABSSQ( A );
            real_t g2 = ABSSQ( B );
            real_t h2 = f2 + g2;
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            C  = f2*p;
            S  = conj( B )*( A*p );
            A *= h2*p ;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, f1, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = B*uu;
            real_t g2 = ABSSQ( gs );
            real_t f2, h2, w;
            scalar_t fs;
            if ( f1*uu < rtmin ) {
                // a is not well-scaled when scaled by g1.
                real_t v = min( safmax, max( safmin, f1 ) );
                real_t vv = r_one / v;
                w = v * uu;
                fs = A*vv;
                f2 = ABSSQ( fs );
                h2 = f2*w*w + g2;
            }
            else {
                // Otherwise use the same scaling for a and b.
                w = r_one;
                fs = A*uu;
                f2 = ABSSQ( fs );
                h2 = f2 + g2;
            }
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            C = ( f2*p )*w;
            S = conj( gs )*( fs*p );
            A = ( fs*( h2*p ) )*u;
        }
    }

    #undef ABSSQ
    #undef A
    #undef B
    #undef C
    #undef S
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTG_HH
