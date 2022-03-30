/// @file lange.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lange.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
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
template< typename norm_t, typename array_t >
auto
lange( norm_t normType, const array_t& a )
{
    using T      = type_t< array_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< array_t >;
    using blas::isnan;
    using blas::sqrt;

    const auto& A = interpretAsMatrix( a );

    // constants
    const real_t rzero(0.0);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    blas_error_if(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    blas_error_if(  access_denied( dense, read_policy(A) ) );

    // quick return
    if (m == 0 || n == 0)
        return rzero;

    // Norm value
    real_t norm = rzero;

    if( normType == Norm::Max )
    {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < m; ++i)
            {
                real_t temp = blas::abs( A(i,j) );

                if (temp > norm)
                    norm = temp;
                else {
                    if ( isnan(temp) ) 
                        return temp;
                }
            }
        }
    }
    else if ( normType == Norm::Inf )
    {
        for (idx_t i = 0; i < m; ++i)
        {
            real_t sum = rzero;
            for (idx_t j = 0; j < n; ++j)
                sum += blas::abs( A(i,j) );

            if (sum > norm)
                norm = sum;
            else {
                if ( isnan(sum) ) 
                    return sum;
            }
        }
    }
    else if ( normType == Norm::One )
    {
        for (idx_t j = 0; j < n; ++j)
        {
            real_t sum = rzero;
            for (idx_t i = 0; i < m; ++i)
                sum += blas::abs( A(i,j) );

            if (sum > norm)
                norm = sum;
            else {
                if ( isnan(sum) ) 
                    return sum;
            }
        }
    }
    else
    {
        real_t scale(0.0), sum(1.0);
        for (idx_t j = 0; j < n; ++j)
            lassq( col(A,j), scale, sum );
        norm = scale * sqrt(sum);
    }

    return norm;
}

template< typename norm_t, typename matrix_t, class work_t >
real_type< type_t< matrix_t > >
lange( norm_t normType, const matrix_t& A, work_t& work )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
    using blas::isnan;

    // check arguments
    blas_error_if(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    blas_error_if(  access_denied( dense, read_policy(A) ) );

    // redirect for max-norm, one-norm and Frobenius norm
    if      ( normType == Norm::Max  ) return lange( max_norm,  A );
    else if ( normType == Norm::One  ) return lange( one_norm,  A );
    else if ( normType == Norm::Fro  ) return lange( frob_norm, A );
    else if ( normType == Norm::Inf  ) {

        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
	    // the infinite norm
 
        // constants
        const real_t rzero(0.0);
        const idx_t m = nrows(A);
        const idx_t n = ncols(A);

        // quick return
        if (m == 0 || n == 0)
            return rzero;

        // Norm value
        real_t norm = rzero;

        for (idx_t i = 0; i < m; ++i)
            work[i] = blas::abs( A(i,0) );
    
        for (idx_t j = 1; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work[i] += blas::abs( A(i,j) );

        for (idx_t i = 0; i < m; ++i)
        {
            real_t temp = work[i];

            if (temp > norm)
                norm = temp;
            else {
                if (isnan(temp))
                    return temp;
            }
        }

        return norm;

    }
}

} // lapack

#endif // __LANGE_HH__
