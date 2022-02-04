/// @file lansy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lansy.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LANSY_HH__
#define __LANSY_HH__

#include "lapack/types.hpp"
#include "lapack/lassq.hpp"

namespace lapack {

/** Calculates the value of the one norm, Frobenius norm, infinity norm, or element of largest absolute value of a symmetric matrix
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
 * @param uplo Indicates whether the symmetric matrix A is stored as upper triangular or lower triangular.
 *      The other triangular part of A is not referenced.
 * @param n Number of columns to be included in the norm. n >= 0
 * @param A symmetric matrix size lda-by-n.
 * @param lda Leading dimension of matrix A.  ldA >= m
 * 
 * @ingroup auxiliary
**/
template< class norm_t, class uplo_t, class matrix_t,
    enable_if_t<
    /* Requires: */
    (   is_same_v<norm_t,max_norm_t> || 
        is_same_v<norm_t,one_norm_t> || 
        is_same_v<norm_t,inf_norm_t> || 
        is_same_v<norm_t,frob_norm_t>
    ) && (
        is_same_v< uplo_t, upper_triangle_t > || 
        is_same_v< uplo_t, lower_triangle_t >
    ), bool > = true
>
real_type< type_t<matrix_t> >
lansy( norm_t normType, uplo_t uplo, const matrix_t& A )
{
    using real_t = real_type< type_t<matrix_t> >;
    using idx_t  = size_type< matrix_t >;
    using pair   = std::pair<idx_t,idx_t>;
    using blas::isnan;
    using blas::sqrt;

    // constants
    const real_t zero(0.0);
    const idx_t n = nrows(A);

    // quick return
    if ( n <= 0 ) return zero;

    // Norm value
    real_t norm(0.0);

    if( is_same_v<norm_t,max_norm_t> )
    {
        if( is_same_v<uplo_t,upper_triangle_t> ) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i <= j; ++i)
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
        else {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = j; i < n; ++i)
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
    }
    else if (
        is_same_v<norm_t,one_norm_t> || 
        is_same_v<norm_t,inf_norm_t> )
    {
        real_t *work = new real_t[n];
        for (idx_t i = 0; i < n; ++i)
            work[i] = zero;

        if( is_same_v<uplo_t,upper_triangle_t> ) {
            for (idx_t j = 0; j < n; ++j)
            {
                real_t sum = zero;
                for (idx_t i = 0; i < j; ++i) {
                    const real_t absa = blas::abs( A(i,j) );
                    sum += absa;
                    work[i] += absa;
                }
                work[j] = sum + blas::abs( A(j,j) );
            }
            for (idx_t i = 0; i < n; ++i)
            {
                real_t sum = work[i];
                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) {
                        delete[] work;
                        return sum;
                    }
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j)
            {
                real_t sum = work[j] + blas::abs( A(j,j) );
                for (idx_t i = j+1; i < n; ++i) {
                    const real_t absa = blas::abs( A(i,j) );
                    sum += absa;
                    work[i] += absa;
                }
                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) {
                        delete[] work;
                        return sum;
                    }
                }
            }
        }
        delete[] work;
    }
    else if ( is_same_v<norm_t,frob_norm_t> )
    {
        // Scaled ssq
        real_t scale(0), ssq(1);
        
        // Sum off-diagonals
        if( is_same_v<uplo_t,upper_triangle_t> ) {
            for (idx_t j = 1; j < n; ++j)
                lassq( subvector( col(A,j), pair{0,j} ), scale, ssq );
        }
        else {
            for (idx_t j = 0; j < n-1; ++j)
                lassq( subvector( col(A,j), pair{j+1,n} ), scale, ssq );
        }
        ssq *= 2;

        // Sum diagonal
        lassq( diag(A,0), scale, ssq );

        // Compute the scaled square root
        norm = scale * sqrt(ssq);
    }

    return norm;
}

} // lapack

#endif // __LANSY_HH__
