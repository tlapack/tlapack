/// @file larfg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larfg.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARFG_HH__
#define __LARFG_HH__

#include "lapack/types.hpp"
#include "lapack/lapy2.hpp"
#include "lapack/lapy3.hpp"
#include "lapack/utils.hpp"

#include "tblas.hpp"

namespace lapack {

/** Generates a elementary Householder reflection.
 *
 * larfg generates a elementary Householder reflection H of order n, such that
 * 
 *        H * ( alpha ) = ( beta ),   H' * H = I.
 *            (   x   )   (   0  )
 * 
 * where alpha and beta are scalars, with beta real, and x is an (n-1)-element vector.
 * H is represented in the form
 * 
 *        H = I - tau * ( 1 ) * ( 1 v' ) 
 *                      ( v )
 * 
 * where tau is a scalar and v is a (n-1)-element vector.
 * Note that H is symmetric but not hermitian.
 * 
 * If the elements of x are all zero and alpha is real, then tau = 0
 * and H is taken to be the identity matrix.
 * 
 * Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
 * 
 * @param[in,out] alpha
 *      On entry, the value alpha.
 *      On exit, it is overwritten with the value beta.
 * 
 * @param[in,out] x Vector of length n-1.
 *      On entry, the vector x.
 *      On exit, it is overwritten with the vector v.
 * 
 * @param[out] tau The value tau.
 * 
 * @ingroup auxiliary
 */
template< class vector_t, class alpha_t, class tau_t >
void larfg( alpha_t& alpha, vector_t& x, tau_t& tau )
{
    // data traits
    using TX    = type_t< vector_t >;
    using idx_t = size_type< vector_t >;

    // using
    using real_t = real_type< alpha_t, TX >;
    using blas::real;
    using blas::imag;
    using blas::abs;
    using blas::nrm2;
    using blas::scal;
    using blas::safe_min;
    using blas::uroundoff;
    using blas::is_complex;

    // constants
    const idx_t n = size(x) + 1;
    const real_t one( 1 );
    const real_t rzero( 0 );
    const real_t safemin  = safe_min<real_t>() / uroundoff<real_t>();
    const real_t rsafemin = one / safemin;

    tau = tau_t( 0 );
    if (n > 0)
    {
        idx_t knt = 0;
        real_t xnorm = nrm2( x );
        if ( xnorm > rzero || (imag(alpha) != rzero) )
        {
            real_t temp = ( ! is_complex<alpha_t>::value )
                        ? lapy2(real(alpha), xnorm)
                        : lapy3(real(alpha), imag(alpha), xnorm);
            real_t beta = (real(alpha) < rzero) ? temp : -temp;
            if (abs(beta) < safemin)
            {
                while( (abs(beta) < safemin) && (knt < 20) )
                {
                    knt++;
                    scal( rsafemin, x );
                    beta *= rsafemin;
                    alpha *= rsafemin;
                }
                xnorm = nrm2( x );
                temp = ( ! is_complex<alpha_t>::value )
                     ? lapy2(real(alpha), xnorm)
                     : lapy3(real(alpha), imag(alpha), xnorm);
                beta = (real(alpha) < rzero) ? temp : -temp;
            }
            tau = (beta - alpha) / beta;
            scal( one/(alpha-beta), x );
            for (idx_t j = 0; j < knt; ++j)
                beta *= safemin;
            alpha = beta;
        }
    }
}

/** Generates a elementary Householder reflection.
 *
 * larfg generates a elementary Householder reflection H of order n, such that
 * 
 *        H * ( alpha ) = ( beta ),   H' * H = I.
 *            (   x   )   (   0  )
 * 
 * where alpha and beta are scalars, with beta real, and x is an (n-1)-element vector.
 * H is represented in the form
 * 
 *        H = I - tau * ( 1 ) * ( 1 v' ) 
 *                      ( v )
 * 
 * where tau is a scalar and v is a (n-1)-element vector.
 * Note that H is symmetric but not hermitian.
 * 
 * If the elements of x are all zero and alpha is real, then tau = 0
 * and H is taken to be the identity matrix.
 * 
 * Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
 * 
 * @param[in,out] v Array of length n.  On entry, the vector (alpha, x).  On exit, it is overwritten with the vector (beta, v).
 * @param[out] tau On exit, the value tau.
 * 
 * @ingroup auxiliary
 */
template< class vector_t, class tau_t >
void larfg( vector_t& v, tau_t& tau )
{
    using idx_t = size_type< vector_t >;
    using pair  = std::pair<idx_t,idx_t>;

    const idx_t n = size(v);
    auto x = slice( v, n > 1 ? pair{1,n} : pair{0,0} );
    larfg( v[0], x, tau );
}

}

#endif // __LARFG_HH__
