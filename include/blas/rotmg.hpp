// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTMG_HH
#define BLAS_ROTMG_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Construct modified (fast) plane rotation, H, that eliminates b, such that
 * \[
 *       \begin{bmatrix} z \\ 0 \end{bmatrix}
 *     = H
 *       \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
 *       \begin{bmatrix} a \\ b \end{bmatrix}.
 * \]
 *
 * @see rotm to apply the rotation.
 *
 * With modified plane rotations, vectors u and v are held in factored form as
 * \[
 *     \begin{bmatrix} u^T \\ v^T \end{bmatrix} =
 *     \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
 *     \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
 * \]
 *
 * Application of H to vectors x and y requires 4n flops (2n mul, 2n add)
 * instead of 6n flops (4n mul, 2n add) as in standard plane rotations.
 *
 * Let param = [ flag, $h_{11}, h_{21}, h_{12}, h_{22}$ ].
 * Then H has one of the following forms:
 *
 * - For flag = -1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  h_{12}
 *         \\  h_{21}  &  h_{22}
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = 0,
 *     \[
 *         H = \begin{bmatrix}
 *             1       &  h_{12}
 *         \\  h_{21}  &  1
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = 1,
 *     \[
 *         H = \begin{bmatrix}
 *             h_{11}  &  1
 *         \\  -1      &  h_{22}
 *         \end{bmatrix}
 *     \]
 *
 * - For flag = -2,
 *     \[
 *         H = \begin{bmatrix}
 *             1  &  0
 *         \\  0  &  1
 *         \end{bmatrix}
 *     \]
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in, out] d1
 *     sqrt(d1) is scaling factor for vector x.
 *
 * @param[in, out] d2
 *     sqrt(d2) is scaling factor for vector y.
 *
 * @param[in, out] a
 *     On entry, scalar a. On exit, set to z.
 *
 * @param[in] b
 *     On entry, scalar b.
 *
 * @param[out] param
 *     Array of length 5 giving parameters of modified plane rotation,
 *     as described above.
 *
 * __Further details__
 *
 * Hammarling, Sven. A note on modifications to the Givens plane rotation.
 * IMA Journal of Applied Mathematics, 13:215-218, 1974.
 * http://dx.doi.org/10.1093/imamat/13.2.215
 * (Note the notation swaps u <=> x, v <=> y, d_i -> l_i.)
 *
 * @ingroup rotmg
 */
template< typename real_t >
void rotmg(
    real_t *d1,
    real_t *d2,
    real_t *a,
    real_t  b,
    real_t param[5] )
{
    #define D1 (*d1)
    #define D2 (*d2)
    #define A  (*a)

    // check arguments
    blas_error_if( D1 <= 0 );

    // Constants
    const real_t zero( 0 );
    const real_t one( 1 );
    const real_t gam = 4096;
    const real_t gamsq = gam*gam;
    const real_t rgamsq = one/gamsq;

    real_t& x1 = A;
    real_t& y1 = b;

    real_t h11 = zero;
    real_t h12 = zero;
    real_t h21 = zero;
    real_t h22 = zero;

    if(D1 < zero) {
        param[0] = -1;
        D1 = zero;
        D2 = zero;
        x1 = zero;
    }
    else {
        real_t p2 = D2*y1;
        if(p2 == zero) {
            param[0] = -2;
            return;
        }

        real_t p1 = D1*x1;
        real_t q2 = p2*y1;
        real_t q1 = p1*x1;

        if( blas::abs(q1) > blas::abs(q2) ) {
            param[0] = zero;
            h21 = -y1/x1;
            h12 = p2/p1;
            real_t u = one - h12*h21;
            if( u > zero ) {
                D1 /= u;
                D2 /= u;
                x1 *= u;
            }
        }
        else if(q2 < zero) {
            param[0] = -1;
            D1 = zero;
            D2 = zero;
            x1 = zero;
        }
        else {
            param[0] = 1;
            h11 = p1/p2;
            h22 = x1/y1;
            real_t u = one + h11*h22;
            real_t stemp = D2/u;
            D2 = D1/u;
            D1 = stemp;
            x1 = y1*u;
        }

        if(D1 != zero) {
            while( (D1 <= rgamsq) || (D1 >= gamsq) ) {
                if(param[0] == 0) {
                    h11 = one;
                    h22 = one;
                    param[0] = -1;
                }
                else {
                    h21 = -one;
                    h12 = one;
                    param[0] = -1;
                }
                if(D1 <= rgamsq) {
                    D1  *= gam*gam;
                    x1  /= gam;
                    h11 /= gam;
                    h12 /= gam;
                }
                else {
                    D1 /= gam*gam;
                    x1  *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }

        if(D2 != zero) {
            while( (blas::abs(D2) <= rgamsq) || (blas::abs(D2) >= gamsq) ) {
                if(param[0] == 0) {
                    h11=one;
                    h22=one;
                    param[0]=-1;
                }
                else {
                    h21=-one;
                    h12=one;
                    param[0]=-1;
                }
                if(blas::abs(D2) <= rgamsq) {
                    D2  *= gam*gam;
                    h21 /= gam;
                    h22 /= gam;
                }
                else {
                    D2 /= gam*gam;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }

    if(param[0] < 0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    }
    else if(param[0] == 0) {
        param[2] = h21;
        param[3] = h12;
    }
    else {
        param[1] = h11;
        param[4] = h22;
    }

    #undef D1
    #undef D2
    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTMG_HH
