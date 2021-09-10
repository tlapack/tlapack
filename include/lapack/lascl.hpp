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
template< typename T >
int lascl(
    lapack::MatrixType matrixtype,
    blas::size_t kl, blas::size_t ku,
    const real_type<T>& b, const real_type<T>& a,
    blas::size_t m, blas::size_t n,
    T* A, blas::size_t lda )
{
    typedef real_type<T> real_t;
    using blas::isnan;
    using blas::max;
    using blas::min;
    using blas::safe_min;

    // constants
    const blas::size_t izero = 0;
    const real_t zero = 0.0;
    const real_t one(1.0);
    const real_t small = safe_min<real_t>();
    const real_t big = one / small;
    
    // check arguments
    lapack_error_if(
        (matrixtype != MatrixType::General) && 
        (matrixtype != MatrixType::Lower) && 
        (matrixtype != MatrixType::Upper) && 
        (matrixtype != MatrixType::Hessenberg) && 
        (matrixtype != MatrixType::LowerBand) && 
        (matrixtype != MatrixType::UpperBand) && 
        (matrixtype != MatrixType::Band), -1 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (kl < 0) ||
            (kl > max(m-1, izero))
        ), -2 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand) || 
            (matrixtype == MatrixType::Band)
        ) && (
            (ku < 0) ||
            (ku > max(n-1, izero))
        ), -3 );
    lapack_error_if( (
            (matrixtype == MatrixType::LowerBand) ||
            (matrixtype == MatrixType::UpperBand)
        ) && ( kl != ku ), -3 );
    lapack_error_if( (b == zero) || isnan(b), -4 );
    lapack_error_if( isnan(a), -5 );
    lapack_error_if( m < 0, -6 );
    lapack_error_if(
        (n < 0) ||
        ((matrixtype == MatrixType::LowerBand) && (n != m)) || 
        ((matrixtype == MatrixType::UpperBand) && (n != m)), -7 );
    lapack_error_if( (lda < m) && (
        (matrixtype == MatrixType::General) || 
        (matrixtype == MatrixType::Lower) ||
        (matrixtype == MatrixType::Upper) ||
        (matrixtype == MatrixType::Hessenberg) ), -9 );
    lapack_error_if( (matrixtype == MatrixType::LowerBand) && (lda < kl + 1), -9);
    lapack_error_if( (matrixtype == MatrixType::UpperBand) && (lda < ku + 1), -9);
    lapack_error_if( (matrixtype == MatrixType::Band) && (lda < 2 * kl + ku + 1), -9);

    #define _A(_i,_j) A[ _i + _j * lda ]

    bool done = 0;
    while (!done)
    {
        real_t a1, c;
        real_t b1 = b * small;
        if (b1 == b)
        {
            c = a / b;
            done = 1;
            a1 = a;
        }
        else
        {
            a1 = a / big;
            if (a1 == a)
            {
                c = a;
                done = 1;
                b = one;
            }
            else if ((abs(b1) > abs(a)) && (a != zero))
            {
                c = small;
                done = 0;
                b = b1;
            }
            else if (abs(a1) > abs(b))
            {
                c = big;
                done = 0;
                a = a1;
            }
            else
            {
                c = a / b;
                done = 1;
            }
        }

        if (matrixtype == MatrixType::General)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = 0; i < m; ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::Lower)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = j; i < m; ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::Upper)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = 0; (i < m) && (i <= j); ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::Hessenberg)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = 0; (i < m) && (i <= j + 1); ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::LowerBand)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = 0; (i <= kl) && (i < n - j); ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::UpperBand)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = max(ku - j, izero); i <= ku; ++i)
                    _A(i,j) *= c;
        }
        else if (matrixtype == MatrixType::Band)
        {
            for (blas::size_t j = 0; j < n; ++j)
                for (blas::size_t i = max(kl + ku - j, kl); i <= min(2 * kl + ku, kl + ku + m - j); ++i)
                    _A(i,j) *= c;
        }
    }
    return 0;

    #undef _A
}

}

#endif // __LASCL_HH__
