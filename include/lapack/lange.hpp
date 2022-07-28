/// @file lange.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lange.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LANGE_HH
#define TLAPACK_LANGE_HH

#include "base/types.hpp"
#include "lapack/lassq.hpp"

namespace tlapack {

/** Calculates the norm of a matrix.
 * 
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] A m-by-n matrix.
 * 
 * @ingroup auxiliary
 */
template< typename norm_t, typename matrix_t >
auto
lange( norm_t normType, const matrix_t& A )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    tlapack_check_false(  access_denied( dense, read_policy(A) ) );

    // quick return
    if (m == 0 || n == 0)
        return real_t( 0 );

    // Norm value
    real_t norm( 0 );

    if( normType == Norm::Max )
    {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < m; ++i)
            {
                real_t temp = tlapack::abs( A(i,j) );

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
            real_t sum( 0 );
            for (idx_t j = 0; j < n; ++j)
                sum += tlapack::abs( A(i,j) );

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
            real_t sum( 0 );
            for (idx_t i = 0; i < m; ++i)
                sum += tlapack::abs( A(i,j) );

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
        real_t scale(0), sum(1);
        for (idx_t j = 0; j < n; ++j)
            lassq( col(A,j), scale, sum );
        norm = scale * sqrt(sum);
    }

    return norm;
}

/** Calculates the norm of a matrix.
 * 
 * Code optimized for the infinity norm on column-major layouts using a workspace
 * of size at least m, where m is the number of rows of A.
 * @see lange( norm_t normType, const matrix_t& A ).
 * 
 * @param work Vector of size at least m.
 * 
 * @ingroup auxiliary
 */
template< typename norm_t, typename matrix_t, class work_t >
real_type< type_t< matrix_t > >
lange( norm_t normType, const matrix_t& A, work_t& work )
{
    using T      = type_t< matrix_t >;
    using real_t = real_type< T >;
    using idx_t  = size_type< matrix_t >;
        
    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(  normType != Norm::Fro &&
                    normType != Norm::Inf &&
                    normType != Norm::Max &&
                    normType != Norm::One );
    tlapack_check_false(  access_denied( dense, read_policy(A) ) );

    // quick return
    if (m == 0 || n == 0)
        return real_t( 0 );

    // redirect for max-norm, one-norm and Frobenius norm
    if      ( normType == Norm::Max  ) return lange( max_norm,  A );
    else if ( normType == Norm::One  ) return lange( one_norm,  A );
    else if ( normType == Norm::Fro  ) return lange( frob_norm, A );
    else if ( normType == Norm::Inf  ) {

        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
	    // the infinite norm.

        // Norm value
        real_t norm( 0 );

        for (idx_t i = 0; i < m; ++i)
            work[i] = tlapack::abs( A(i,0) );
    
        for (idx_t j = 1; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work[i] += tlapack::abs( A(i,j) );

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

#endif // TLAPACK_LANGE_HH
