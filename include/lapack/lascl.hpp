/// @file lascl.hpp Multiplies a matrix by a scalar.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lascl.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LASCL_HH__
#define __LASCL_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

namespace lapack {

/**
 * @brief Multiplies a matrix A by the real scalar a/b.
 *
 * Multiplication of a matrix A by scalar a/b is done without over/underflow as
 * long as the final result $a A/b$ does not over/underflow.
 * 
 * @tparam access_t Type of access inside the algorithm.
 *      Either MatrixAccessPolicy or any type that implements
 *          operator MatrixAccessPolicy().
 * @tparam matrix_t Matrix type.
 * @tparam a_type Type of the coefficient a.
 *      a_type cannot be a complex type.
 * @tparam b_type Type of the coefficient b.
 *      b_type cannot be a complex type.
 * 
 * @param[in] accessType Determines the entries of A that are scaled by a/b.
 *      The following access types are allowed:
 *          MatrixAccessPolicy::Dense,
 *          MatrixAccessPolicy::UpperHessenberg,
 *          MatrixAccessPolicy::LowerHessenberg,
 *          MatrixAccessPolicy::UpperTriangle,
 *          MatrixAccessPolicy::LowerTriangle,
 *          MatrixAccessPolicy::StrictUpper,
 *          MatrixAccessPolicy::StrictLower.
 * 
 * @param[in] b The denominator of the scalar a/b.
 * @param[in] a The numerator of the scalar a/b.
 * @param[in,out] A Matrix to be scaled by a/b.
 * 
 * @return  0 if success.
 * @return -i if the ith argument is invalid.
 * 
 * @ingroup auxiliary
 */
template< class access_t, class matrix_t, class a_type, class b_type,
    enable_if_t<(
    /* Requires: */
        !is_complex< a_type >::value &&
        !is_complex< b_type >::value
    ), int > = 0 >
int lascl(
    access_t accessType,
    const b_type& b, const a_type& a,
    const matrix_t& A )
{
    // data traits
    using idx_t  = size_type< matrix_t >;
    using real_t = real_type< a_type, b_type >;
    
    // using
    using blas::isnan;
    using blas::abs;
    using blas::safe_min;
    using blas::safe_max;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // constants
    const real_t small = safe_min<real_t>();
    const real_t big   = safe_max<real_t>();
    
    // check arguments
    lapack_error_if(
        (accessType != MatrixAccessPolicy::Dense) && 
        (accessType != MatrixAccessPolicy::UpperHessenberg) && 
        (accessType != MatrixAccessPolicy::LowerHessenberg) && 
        (accessType != MatrixAccessPolicy::UpperTriangle) && 
        (accessType != MatrixAccessPolicy::LowerTriangle) && 
        (accessType != MatrixAccessPolicy::StrictUpper) && 
        (accessType != MatrixAccessPolicy::StrictLower), -1 );
    lapack_error_if( access_denied( accessType, write_policy(A) ), -1 );
    lapack_error_if( (b == b_type(0)) || isnan(b), -2 );
    lapack_error_if( isnan(a), -3 );

    // quick return
    if( m <= 0 || n <= 0 )
        return 0;

    bool done = false;
    real_t _a = a, _b = b;
    while (!done)
    {
        real_t c, a1, b1 = b * small;
        if (b1 == _b) {
            // b is not finite:
            //  c is a correctly signed zero if a is finite,
            //  c is NaN otherwise.
            c = _a / _b;
            done = true;
        }
        else { // b is finite
            a1 = _a / big;
            if (a1 == _a) {
                // a is either 0 or an infinity number:
                //  in both cases, c = a serves as the correct multiplication factor.
                c = _a;
                done = true;
            }
            else if ( (abs(b1) > abs(_a)) && (_a != real_t(0)) ) {
                // a is a non-zero finite number and abs(a/b) < small:
                //  Set c = small as the multiplication factor,
                //  Multiply b by the small factor.
                c = small;
                done = false;
                _b = b1;
            }
            else if (abs(a1) > abs(_b)) {
                // abs(a/b) > big:
                //  Set c = big as the multiplication factor,
                //  Divide a by the big factor.
                c = big;
                done = false;
                _a = a1;
            }
            else {
                // small <= abs(a/b) <= big:
                //  Set c = a/b as the multiplication factor.
                c = _a / _b;
                done = true;
            }
        }

        if ( accessType == MatrixAccessPolicy::UpperHessenberg )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j+2 : m); ++i)
                    A(i,j) *= c;
        }
        else if ( accessType == MatrixAccessPolicy::UpperTriangle )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j+1 : m); ++i)
                    A(i,j) *= c;
        }
        else if ( accessType == MatrixAccessPolicy::StrictUpper )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                    A(i,j) *= c;
        }
        else if ( accessType == MatrixAccessPolicy::LowerHessenberg )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = ((j > 1) ? j-1 : 0); i < m; ++i)
                    A(i,j) *= c;
        }
        else if ( accessType == MatrixAccessPolicy::LowerTriangle )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j; i < m; ++i)
                    A(i,j) *= c;
        }
        else if ( accessType == MatrixAccessPolicy::StrictLower )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j+1; i < m; ++i)
                    A(i,j) *= c;
        }
        else // if ( accessType == MatrixAccessPolicy::Dense )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    A(i,j) *= c;
        }
    }

    return 0;
}

/**
 * @brief Multiplies a matrix A by the real scalar a/b.
 * 
 * Specific implementation for band access types.
 * 
 * @param[in] accessType Determines the entries of A that are scaled by a/b.
 * 
 * @see lascl(
    access_t accessType,
    const b_type& b, const a_type& a,
    const matrix_t& A )
 * 
 * @ingroup auxiliary
 */
template< class matrix_t, class a_type, class b_type,
    enable_if_t<(
    /* Requires: */
        !is_complex< a_type >::value &&
        !is_complex< b_type >::value
    ), int > = 0 >
int lascl(
    band_t accessType,
    const b_type& b, const a_type& a,
    const matrix_t& A )
{
    // data traits
    using idx_t  = size_type< matrix_t >;
    using real_t = real_type< a_type, b_type >;
    
    // using
    using blas::isnan;
    using blas::abs;
    using blas::safe_min;
    using blas::safe_max;
    using std::min;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    // constants
    const real_t small = safe_min<real_t>();
    const real_t big   = safe_max<real_t>();
    
    // check arguments
    lapack_error_if( (kl < 0) || (kl >= m) || (ku < 0) || (ku >= n), -1 );
    lapack_error_if( access_denied( accessType, write_policy(A) ), -1 );
    lapack_error_if( (b == b_type(0)) || isnan(b), -2 );
    lapack_error_if( isnan(a), -3 );

    // quick return
    if( m <= 0 || n <= 0 )
        return 0;

    bool done = false;
    real_t _a = a, _b = b;
    while (!done)
    {
        real_t c, a1, b1 = b * small;
        if (b1 == _b) {
            // b is not finite:
            //  c is a correctly signed zero if a is finite,
            //  c is NaN otherwise.
            c = _a / _b;
            done = true;
        }
        else { // b is finite
            a1 = _a / big;
            if (a1 == _a) {
                // a is either 0 or an infinity number:
                //  in both cases, c = a serves as the correct multiplication factor.
                c = _a;
                done = true;
            }
            else if ( (abs(b1) > abs(_a)) && (_a != real_t(0)) ) {
                // a is a non-zero finite number and abs(a/b) < small:
                //  Set c = small as the multiplication factor,
                //  Multiply b by the small factor.
                c = small;
                done = false;
                _b = b1;
            }
            else if (abs(a1) > abs(_b)) {
                // abs(a/b) > big:
                //  Set c = big as the multiplication factor,
                //  Divide a by the big factor.
                c = big;
                done = false;
                _a = a1;
            }
            else {
                // small <= abs(a/b) <= big:
                //  Set c = a/b as the multiplication factor.
                c = _a / _b;
                done = true;
            }
        }

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j >= ku) ? (j-ku) : 0); i < min(m,j+kl+1); ++i)
                A(i,j) *= c;
    }

    return 0;
}

}

#endif // __LASCL_HH__
