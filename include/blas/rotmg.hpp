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
int rotmg(
    real_t& d1, real_t& d2,
    real_t& a, const real_t& b,
    real_t H[4] )
{
    // check arguments
    blas_error_if( d1 <= 0 );

    // Constants
    const real_t zero( 0 );
    const real_t one( 1 );
    const real_t gam( 4096 );
    const auto gamsq  = gam*gam;
    const auto rgamsq = one/gamsq;

    int flag;
    H[0] = zero;
    H[1] = zero;
    H[2] = zero;
    H[3] = zero;

    if(d1 < zero) {
        flag = -1;
        d1 = zero;
        d2 = zero;
        a = zero;
    }
    else {
        auto p2 = d2*b;
        if(p2 == zero) {
            flag = -2;
        }
        else {
            auto p1 = d1*a;
            auto q2 = p2*b;
            auto q1 = p1*a;

            if( blas::abs(q1) > blas::abs(q2) ) {
                flag = zero;
                H[1] = -b/a;
                H[2] = p2/p1;
                auto u = one - H[2]*H[1];
                if( u > zero ) {
                    d1 /= u;
                    d2 /= u;
                    a *= u;
                }
            }
            else if(q2 < zero) {
                flag = -1;
                d1 = zero;
                d2 = zero;
                a = zero;
            }
            else {
                flag = 1;
                H[0] = p1/p2;
                H[3] = a/b;
                auto u = one + H[0]*H[3];
                auto stemp = d2/u;
                d2 = d1/u;
                d1 = stemp;
                a = b*u;
            }

            if(d1 != zero) {
                while( (d1 <= rgamsq) || (d1 >= gamsq) ) {
                    if(flag == 0) {
                        H[0] = one;
                        H[3] = one;
                        flag = -1;
                    }
                    else {
                        H[1] = -one;
                        H[2] = one;
                        flag = -1;
                    }
                    if(d1 <= rgamsq) {
                        d1  *= gam*gam;
                        a  /= gam;
                        H[0] /= gam;
                        H[2] /= gam;
                    }
                    else {
                        d1 /= gam*gam;
                        a  *= gam;
                        H[0] *= gam;
                        H[2] *= gam;
                    }
                }
            }

            if(d2 != zero) {
                while( (blas::abs(d2) <= rgamsq) || (blas::abs(d2) >= gamsq) ) {
                    if(flag == 0) {
                        H[0]=one;
                        H[3]=one;
                        flag=-1;
                    }
                    else {
                        H[1]=-one;
                        H[2]=one;
                        flag=-1;
                    }
                    if(blas::abs(d2) <= rgamsq) {
                        d2  *= gam*gam;
                        H[1] /= gam;
                        H[3] /= gam;
                    }
                    else {
                        d2 /= gam*gam;
                        H[1] *= gam;
                        H[3] *= gam;
                    }
                }
            }
        }
    }

    return flag;
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTMG_HH
