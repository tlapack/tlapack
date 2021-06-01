// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/larfg.h
/// @author Rodney James, University of Colorado Denver, USA

#ifndef __LARFG_HH__
#define __LARFG_HH__

#include "lapack/types.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/ladiv.hpp"

#include "lapack/utils.hpp"
#include "blas/nrm2.hpp"
#include "blas/scal.hpp"

namespace lapack {

/** Generates a real elementary Householder reflection.
 *
 * The real version of larfg generates a real elementary Householder reflection H of order n, such that
 * 
 *        H * ( alpha ) = ( beta ),   H' * H = I.
 *            (   x   )   (   0  )
 * 
 * where alpha and beta are scalars, and x is an (n-1)-element vector. H is represented in the form
 * 
 *        H = I - tau * ( 1 ) * ( 1 v' ) 
 *                      ( v )
 * 
 * where tau is a real scalar and v is a real (n-1)-element vector.
 * Note that H is symmetric.
 * 
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the identity matrix.
 * 
 * Otherwise  1 <= tau <= 2.
 *
 * @tparam real_t Floating-point type.
 * @param[in] n The order of the elementary Householder reflection.
 * @param[in,out] alpha On entry, the value alpha.  On exit, it is overwritten with the value beta.
 * @param[in,out] x Array of length 1+(n-2)*abs(incx).  On entry, the vector x.  On exit, it is overwritten with the vector v.
 * @param[in] incx  The increment between elements of x; incx > 0.
 * @param[out] tau On exit, the value tau.
 * 
 * @ingroup auxiliary
 */
template< typename real_t >
void larfg(
    blas::size_t n, real_t &alpha, real_t *x, blas::int_t incx, real_t &tau )
{
    using blas::real;
    using blas::imag;
    using blas::abs;
    using blas::nrm2;
    using blas::scal;

    // constants
    const real_t one(1.0);
    const real_t zero(0.0);
    const real_t safemin  = blas::safe_min<real_t>();
    const real_t rsafemin = 1.0 / safemin;

    tau = zero;
    if (n > 0)
    {
        int_t knt = 0;
        real_t xnorm = nrm2( n-1, x, incx );
        if ( xnorm > zero )
        {
            real_t temp = abs( lapy2(alpha, xnorm) );
            real_t beta = (alpha < zero) ? temp : -temp;
            if (abs(beta) < safemin)
            {
                while (abs(beta) < safemin)
                {
                    knt++;
                    scal( n-1, rsafemin, x, incx );
                    beta *= rsafemin;
                    alpha *= rsafemin;
                }
                xnorm = nrm2( n-1, x, incx );
                temp = abs( lapy2(alpha, xnorm) );
                beta = (alpha < zero) ? temp : -temp;
            }
            tau = (beta - alpha) / beta;
            scal( n-1, one/(alpha-beta), x, incx );
            for (int_t j = 0; j < knt; ++j)
                beta *= safemin;
            alpha = beta;
        }
    }
}

/** Generates a complex elementary Householder reflection.
 *
 * The complex version of larfg generates a complex elementary Householder reflector H of order n, such that
 * 
 *        H' * ( alpha ) = ( beta ),   H' * H = I.
 *             (   x   )   (   0  )
 * 
 * where alpha and beta are scalars, with beta real, and x is an (n-1)-element complex vector. 
 * H is represented in the form
 * 
 *        H = I - tau * ( 1 ) * ( 1 v' ) 
 *                      ( v )
 * 
 * where tau is a complex scalar and v is a complex (n-1)-element vector.
 * Note that H is not hermitian.
 * 
 * If the elements of x are all zero and alpha is complex, then tau = 0 and 
 * H is taken to be the identity matrix.
 * 
 * Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
 *
 * @tparam real_t Floating-point type.
 * @param[in] n The order of the elementary Householder reflection.
 * @param[in,out] alpha On entry, the value alpha.  On exit, it is overwritten with the value beta.
 * @param[in,out] x Array of length 1+(n-2)*abs(incx).  On entry, the vector x.  On exit, it is overwritten with the vector v.
 * @param[in] incx  The increment between elements of x; incx > 0.
 * @param[out] tau On exit, the value tau.
 * 
 * @ingroup auxiliary
 */
template< typename real_t >
void larfg(
    blas::size_t n, std::complex<real_t> &alpha,
    std::complex<real_t> *x, blas::int_t incx, std::complex<real_t> &tau )
{
    using blas::real;
    using blas::imag;
    using blas::abs;
    using blas::nrm2;
    using blas::scal;

    // constants
    const std::complex<real_t> one(1.0);
    const real_t zero(0.0);
    const real_t safemin  = blas::safe_min<real_t>();
    const real_t rsafemin = 1.0 / safemin;

    tau = std::complex<real_t>( 0, 0 );
    if (n > 0)
    {
        int_t knt = 0;
        real_t xnorm = nrm2( n-1, x, incx );
        if ( (xnorm > zero) || (imag(alpha) != zero) )
        {
            real_t temp = abs( lapy3(real(alpha), imag(alpha), xnorm) );
            real_t beta = (real(alpha) < zero) ? temp : -temp;
            if (abs(beta) < safemin)
            {
                while (abs(beta) < safemin)
                {
                    knt++;
                    scal( n-1, rsafemin, x, incx );
                    beta *= rsafemin;
                    alpha *= rsafemin;
                }
                xnorm = nrm2( n-1, x, incx );
                temp = abs( lapy3(real(alpha), imag(alpha), xnorm) );
                beta = (real(alpha) < zero) ? temp : -temp;
            }
            tau = std::complex<real_t>(
                (beta - real(alpha)) / beta, -imag(alpha) / beta );
            alpha = ladiv( one, alpha - beta );
            scal( n-1, alpha, x, incx );
            for (int_t j = 0; j < knt; ++j)
                beta *= safemin;
            alpha = beta;
        }
    }
}

}

#endif // __LARFG_HH__