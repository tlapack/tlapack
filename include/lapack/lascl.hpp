/// @file lascl.hpp Multiplies a matrix by a scalar.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lascl.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LASCL_HH__
#define __LASCL_HH__

#include "lapack/types.hpp"
#include "lapack/utils.hpp"

namespace lapack {

/** @brief  Multiplies a matrix A by the real scalar a/b.
 *
 * Multiplication of a matrix A by scalar a/b is done without over/underflow as long as the final
 * result $a A/b$ does not over/underflow. The parameter type specifies that
 * A may be full, upper triangular, lower triangular, upper Hessenberg, or banded.
 * 
 * @return 0 if success.
 * @return -i if the ith argument is invalid.
 * 
 * @param[in] type Specifies the type of matrix A.
 *
 *        MatrixType::General: 
 *          A is a full matrix.
 *        MatrixType::Lower:
 *          A is a lower triangular matrix.
 *        MatrixType::Upper:
 *          A is an upper triangular matrix.
 *        MatrixType::Hessenberg:
 *          A is an upper Hessenberg matrix.
 *        MatrixType::LowerBand:
 *          A is a symmetric band matrix with lower bandwidth kl and upper bandwidth ku
 *          and with the only the lower half stored.
 *        MatrixType::UpperBand: 
 *          A is a symmetric band matrix with lower bandwidth kl and upper bandwidth ku
 *          and with the only the upper half stored.
 *        MatrixType::Band:
 *          A is a band matrix with lower bandwidth kl and upper bandwidth ku.
 * 
 * @param[in] kl The lower bandwidth of A, used only for banded matrix types B, Q and Z.
 * @param[in] ku The upper bandwidth of A, used only for banded matrix types B, Q and Z.
 * @param[in] b The denominator of the scalar a/b.
 * @param[in] a The numerator of the scalar a/b.
 * @param[in] m The number of rows of the matrix A. m>=0
 * @param[in] n The number of columns of the matrix A. n>=0
 * @param[in,out] A Pointer to the matrix A [in/out].
 * @param[in] lda The column length of the matrix A.
 * 
 * @ingroup auxiliary
 */
template< class matrix_t, class a_type, class b_type, class sparse_t,
    enable_if_t<((
    /* Requires: */
        is_same_v< sparse_t, general_matrix_t > || 
        is_same_v< sparse_t, lower_triangle_t > || 
        is_same_v< sparse_t, upper_triangle_t > || 
        is_same_v< sparse_t, hessenberg_matrix_t > || 
        is_same_v< sparse_t, symmetric_lowerband_t > || 
        is_same_v< sparse_t, symmetric_upperband_t > || 
        is_same_v< sparse_t, band_matrix_t >
    ) && 
        !is_complex< a_type >::value &&
        !is_complex< b_type >::value
    ), int > = 0
>
int lascl(
    sparse_t matrixtype,
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
    const auto m = nrows(A);
    const auto n = ncols(A);

    // constants
    const idx_t izero = 0;
    const real_t zero = 0.0;
    const real_t one(1.0);
    const real_t small = safe_min<real_t>();
    const real_t big   = safe_max<real_t>();
    
    // check arguments
    if( is_same_v< sparse_t, symmetric_lowerband_t > ||
        is_same_v< sparse_t, symmetric_upperband_t > )
    {
        lapack_error_if((matrixtype.bandwidth + 1 > m && m > 0) ||
                        (matrixtype.bandwidth + 1 > n && n > 0), -1 );
    }
    else if( is_same_v< sparse_t, band_matrix_t > )
    {
        lapack_error_if((matrixtype.lower_bandwidth + 1 > m && m > 0) ||
                        (matrixtype.upper_bandwidth + 1 > n && n > 0), -1 );
    }
    lapack_error_if( (b == b_type(0)) || isnan(b), -2 );
    lapack_error_if( isnan(a), -3 );
    if( is_same_v< sparse_t, symmetric_lowerband_t > ||
        is_same_v< sparse_t, symmetric_upperband_t > )
    {
        lapack_error_if( m != n, -4 );
    }

    // quick return
    if( m <= 0 || n <= 0 )
        return 0;

    bool done = false;
    while (!done)
    {
        real_t c;
        a_type a1;
        b_type b1 = b * small;
        if (b1 == b) {
            // b is not finite:
            //  c is a correctly signed zero if a is finite,
            //  c is NaN otherwise.
            c = a / b;
            done = true;
        }
        else { // b is finite
            a1 = a / big;
            if (a1 == a) {
                // a is either 0 or an infinity number:
                //  in both cases, c = a serves as the correct multiplication factor.
                c = a;
                done = true;
            }
            else if ( (abs(b1) > abs(a)) && (a != a_type(0)) ) {
                // a is a non-zero finite number and abs(a/b) < small:
                //  Set c = small as the multiplication factor,
                //  Multiply b by the small factor.
                c = small;
                done = false;
                b = b1;
            }
            else if (abs(a1) > abs(b)) {
                // abs(a/b) > big:
                //  Set c = big as the multiplication factor,
                //  Divide a by the big factor.
                c = big;
                done = false;
                a = a1;
            }
            else {
                // small <= abs(a/b) <= big:
                //  Set c = a/b as the multiplication factor.
                c = a / b;
                done = true;
            }
        }

        if ( is_same_v< sparse_t, general_matrix_t > )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, lower_triangle_t > )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j; i < m; ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, general_matrix_t > )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; (i < m) && (i <= j); ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, hessenberg_matrix_t > )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; (i < m) && (i <= j + 1); ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, symmetric_lowerband_t > )
        {
            const idx_t k = matrixtype.bandwidth;
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; (i <= k) && (i < n - j); ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, symmetric_upperband_t > )
        {
            const idx_t k = matrixtype.bandwidth;
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = max(k - j, izero); i <= k; ++i)
                    A(i,j) *= c;
        }
        else if ( is_same_v< sparse_t, band_matrix_t > )
        {
            const idx_t kl = matrixtype.lower_bandwidth;
            const idx_t ku = matrixtype.upper_bandwidth;
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = max(kl + ku - j, kl); i <= min(2 * kl + ku, kl + ku + m - j); ++i)
                    A(i,j) *= c;
        }
    }

    return 0;
}

template< typename T >
int lascl(
    lapack::MatrixType matrixtype,
    idx_t kl, idx_t ku,
    const real_type<T>& b, const real_type<T>& a,
    idx_t m, idx_t n,
    T* A, idx_t lda )
{
    using blas::internal::colmajor_matrix;
    using std::max;
    
    // check arguments
    lapack_error_if(
        (matrixtype != MatrixType::General) && 
        (matrixtype != MatrixType::Lower) && 
        (matrixtype != MatrixType::Upper) && 
        (matrixtype != MatrixType::Hessenberg), -1 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (kl < 0) ||
            (kl > max(m-1, idx_t(0)))
        ), -2 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (ku < 0) ||
            (ku > max(n-1, idx_t(0)))
        ), -3 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand)
        ) && ( kl != ku ), -3 );
    lapack_error_if( m < 0, -6 );
    lapack_error_if( (lda < m) && (
        (matrixtype == MatrixType::General) || 
        (matrixtype == MatrixType::Lower) ||
        (matrixtype == MatrixType::Upper) ||
        (matrixtype == MatrixType::Hessenberg) ), -9 );
    lapack_error_if( (matrixtype == MatrixType::LowerBand) && (lda < kl + 1), -9);
    lapack_error_if( (matrixtype == MatrixType::UpperBand) && (lda < ku + 1), -9);
    lapack_error_if( (matrixtype == MatrixType::Band) && (lda < 2 * kl + ku + 1), -9);

    // Matrix views
    auto _A = colmajor_matrix<T>( A, m, n, lda );

    if (matrixtype == MatrixType::General)
        return lascl( general_matrix, b, a, _A );
    else if (matrixtype == MatrixType::Lower)
        return lascl( lower_triangle, b, a, _A );
    else if (matrixtype == MatrixType::Upper)
        return lascl( upper_triangle, b, a, _A );
    else if (matrixtype == MatrixType::Hessenberg)
        return lascl( hessenberg_matrix, b, a, _A );
    else if (matrixtype == MatrixType::LowerBand)
        return lascl( symmetric_lowerband_t{kl}, b, a, _A );
    else if (matrixtype == MatrixType::UpperBand)
        return lascl( symmetric_upperband_t{ku}, b, a, _A );
    else if (matrixtype == MatrixType::Band)
        return lascl( band_matrix_t{kl,ku}, b, a, _A );

    return 0;
}

}

#endif // __LASCL_HH__
