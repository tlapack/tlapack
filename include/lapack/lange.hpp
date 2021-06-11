/// @file lange.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lange.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LANGE_HH__
#define __LANGE_HH__

#include "lapack/types.hpp"
#include "lapack/lassq.hpp"

namespace lapack {

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value
 *
 * @return Calculated norm value for the specified type.
 * 
 * @param normType Type should be specified as follows:
 *
 *     Norm::Max = maximum absolute value over all elements in A.
 *         Note: this is not a consistent matrix norm.
 *     Norm::One = one norm of the matrix A, the maximum value of the sums of each column.
 *     Norm::Inf = the infinity norm of the matrix A, the maximum value of the sum of each row.
 *     Norm::Fro = the Frobenius norm of the matrix A.
 *         This the square root of the sum of the squares of each element in A.
 *
 * @param m Number of rows to be included in the norm. m >= 0
 * @param n Number of columns to be included in the norm. n >= 0
 * @param A matrix size m-by-n.
 * @param ldA Column length of the matrix A.  ldA >= m
 * 
 * @ingroup auxiliary
**/
template <typename TA>
real_type<TA> lange(
    Norm normType, blas::size_t m, blas::size_t n,
    const TA *A, blas::size_t lda )
{
    typedef real_type<TA> real_t;
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    using blas::abs;
    using blas::isnan;
    using blas::sqrt;

    // constants
    const real_t zero(0.0);

    // quick return
    if (m == 0 || n == 0)
        return zero;

    // Norm value
    real_t norm(0.0);

    if( normType == Norm::Max )
    {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i)
            {
                real_t temp = abs( A(i,j) );

                if (temp > norm)
                    norm = temp;
                else {
                    if ( isnan(temp) ) 
                        return temp;
                }
            }
        }
    }
    else if ( normType == Norm::One )
    {
        for (size_t j = 0; j < n; ++j)
        {
            real_t sum = zero;
            for (size_t i = 0; i < m; ++i)
                sum += abs( A(i,j) );

            if (sum > norm)
                norm = sum;
            else {
                if ( isnan(sum) ) 
                    return sum;
            }
        }
    }
    else if ( normType == Norm::Inf )
    {
        real_t *work = new real_t[m];
        for (size_t i = 0; i < m; ++i)
            work[i] = abs( A(i,0) );
        
        for (size_t j = 1; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work[i] += abs( A(i,j) );

        for (size_t i = 0; i < m; ++i)
        {
            real_t temp = work[i];

            if (temp > norm)
                norm = temp;
            else {
                if (isnan(temp)) {
                    delete[] work;
                    return temp;
                }
            }
        }
        delete[] work;
    }
    else if ( normType == Norm::Fro )
    {
        real_t scale(0.0), sum(1.0);
        for (size_t j = 0; j < n; ++j)
            lassq(m, &(A(0,j)), 1, scale, sum);
        norm = scale * sqrt(sum);
    }

    #undef A
    return norm;
}

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value
 * 
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 * @see lange( Norm normType, blas::size_t m, blas::size_t n, const TA *A, blas::size_t lda )
 * 
 * @ingroup auxiliary
**/
template <typename TA>
inline real_type<TA> lange(
    Layout layout,
    Norm normType, blas::size_t m, blas::size_t n,
    const TA *A, blas::size_t lda )
{
    if ( layout == Layout::RowMajor ) {
        
        // Change norm if norm == Norm::One or norm == Norm::Inf
        if( normType == Norm::One ) normType = Norm::Inf;
        else if( normType == Norm::Inf ) normType = Norm::One;
        
        // Transpose A
        std::swap(m,n);
    }

    return lange( normType, m, n, A, lda );
}

} // lapack

#endif // __LANGE_HH__