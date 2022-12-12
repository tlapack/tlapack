// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_ROTMG_HH
#define TLAPACK_BLAS_ROTMG_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Construct modified (fast) plane rotation, H, that eliminates b, such that
 * \[
 *       \begin{bmatrix} z \\ 0 \end{bmatrix}
 *     := H
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
 * Let h = [ $h_{11}, h_{21}, h_{12}, h_{22}$ ].
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
 * @param[in,out] d1 sqrt(d1) is scaling factor for vector x.
 * @param[in,out] d2 sqrt(d2) is scaling factor for vector y.
 * @param[in,out] a On entry, scalar a. On exit, set to z.
 * @param[in]     b Scalar b.
 * @param[out] h 4-element array with the modified plane rotation.
 * \[
 *      h = { h_{11}, h_{21}, h_{12}, h_{22} }.
 * \]
 *
 * @details
 *
 * Hammarling, Sven. A note on modifications to the Givens plane rotation.
 * IMA Journal of Applied Mathematics, 13:215-218, 1974.
 * http://dx.doi.org/10.1093/imamat/13.2.215
 * (Note the notation swaps u <=> x, v <=> y, d_i -> l_i.)
 *
 * @ingroup rotmg
 */
template< typename T,
    enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
    disable_if_allow_optblas_t< T > = 0
>
int rotmg( T& d1, T& d2, T& a, const T& b, T h[4] )
{
    // check arguments
    tlapack_check_false( d1 <= 0 );

    // Constants
    const T zero( 0 );
    const T one( 1 );
    const T gam( 4096 );
    const auto gamsq  = gam*gam;
    const auto rgamsq = one/gamsq;

    int flag;
    h[0] = zero;
    h[1] = zero;
    h[2] = zero;
    h[3] = zero;

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

            if( tlapack::abs(q1) > tlapack::abs(q2) ) {
                flag = zero;
                h[1] = -b/a;
                h[2] = p2/p1;
                auto u = one - h[2]*h[1];
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
                h[0] = p1/p2;
                h[3] = a/b;
                auto u = one + h[0]*h[3];
                auto stemp = d2/u;
                d2 = d1/u;
                d1 = stemp;
                a = b*u;
            }

            if(d1 != zero) {
                while( (d1 <= rgamsq) || (d1 >= gamsq) ) {
                    if(flag == 0) {
                        h[0] = one;
                        h[3] = one;
                        flag = -1;
                    }
                    else {
                        h[1] = -one;
                        h[2] = one;
                        flag = -1;
                    }
                    if(d1 <= rgamsq) {
                        d1  *= gam*gam;
                        a  /= gam;
                        h[0] /= gam;
                        h[2] /= gam;
                    }
                    else {
                        d1 /= gam*gam;
                        a  *= gam;
                        h[0] *= gam;
                        h[2] *= gam;
                    }
                }
            }

            if(d2 != zero) {
                while( (tlapack::abs(d2) <= rgamsq) || (tlapack::abs(d2) >= gamsq) ) {
                    if(flag == 0) {
                        h[0]=one;
                        h[3]=one;
                        flag=-1;
                    }
                    else {
                        h[1]=-one;
                        h[2]=one;
                        flag=-1;
                    }
                    if(tlapack::abs(d2) <= rgamsq) {
                        d2  *= gam*gam;
                        h[1] /= gam;
                        h[3] /= gam;
                    }
                    else {
                        d2 /= gam*gam;
                        h[1] *= gam;
                        h[3] *= gam;
                    }
                }
            }
        }
    }

    return flag;
}

#ifdef USE_LAPACKPP_WRAPPERS

    template< typename T,
        enable_if_t< is_same_v< T, real_type<T> >, int > = 0,
        enable_if_allow_optblas_t< T > = 0
    >
    inline
    int rotmg( T& d1, T& d2, T& a, const T b, T h[4] )
    {
        T param[5];
        ::blas::rotmg( &d1, &d2, &a, b, param );
        
        h[0] = param[1];
        h[1] = param[2];
        h[2] = param[3];
        h[3] = param[4];
        
        return param[0];
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_ROTMG_HH
