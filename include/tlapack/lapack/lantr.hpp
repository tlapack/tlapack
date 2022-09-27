/// @file lantr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lantr.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LANTR_HH
#define TLAPACK_LANTR_HH

#include "tlapack/base/types.hpp"
#include "tlapack/lapack/lassq.hpp"

namespace tlapack {

/** Calculates the norm of a symmetric matrix.
 * 
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * @tparam diag_t Either Diag or any class that implements `operator Diag()`.
 * 
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: A is a upper triangle matrix;
 *      - Uplo::Lower: A is a lower triangle matrix.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 * 
 * @param[in] A m-by-n triangular matrix.
 * 
 * @ingroup auxiliary
 */
template< class norm_t, class uplo_t, class diag_t, class matrix_t >
auto
lantr( norm_t normType, uplo_t uplo, diag_t diag, const matrix_t& A )
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
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );
    tlapack_check_false( diag != Diag::NonUnit &&
                         diag != Diag::Unit );
    tlapack_check_false(  access_denied( uplo, read_policy(A) ) );

    // quick return
    if (m == 0 || n == 0)
        return real_t( 0 );

    // Norm value
    real_t norm( 0 );

    if( normType == Norm::Max )
    {
        if( diag == Diag::NonUnit ) {
            if( uplo == Uplo::Upper ) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= std::min(j,m-1); ++i)
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
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < m; ++i)
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
        }
        else {
            norm = real_t( 1 );
            if( uplo == Uplo::Upper ) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < std::min(j,m); ++i)
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
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j+1; i < m; ++i)
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
        }
    }
    else if( normType == Norm::Inf )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t i = 0; i < m; ++i)
            {
                real_t sum( 0 );
                if( diag == Diag::NonUnit )
                    for (idx_t j = i; j < n; ++j)
                        sum += tlapack::abs( A(i,j) );
                else {
                    sum = real_t( 1 );
                    for (idx_t j = i+1; j < n; ++j)
                        sum += tlapack::abs( A(i,j) );
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) 
                        return sum;
                }
            }
        }
        else {
            for (idx_t i = 0; i < m; ++i)
            {
                real_t sum( 0 );
                if( diag == Diag::NonUnit || i >= n )
                    for (idx_t j = 0; j <= std::min(i,n-1); ++j)
                        sum += tlapack::abs( A(i,j) );
                else {
                    sum = real_t( 1 );
                    for (idx_t j = 0; j < i; ++j)
                        sum += tlapack::abs( A(i,j) );
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) 
                        return sum;
                }
            }
        }
    }
    else if( normType == Norm::One )
    {
        if( uplo == Uplo::Upper ) {
            for (idx_t j = 0; j < n; ++j)
            {
                real_t sum( 0 );
                if( diag == Diag::NonUnit || j >= m )
                    for (idx_t i = 0; i <= std::min(j,m-1); ++i)
                        sum += tlapack::abs( A(i,j) );
                else {
                    sum = real_t( 1 );
                    for (idx_t i = 0; i < j; ++i)
                        sum += tlapack::abs( A(i,j) );
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) 
                        return sum;
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j)
            {
                real_t sum( 0 );
                if( diag == Diag::NonUnit )
                    for (idx_t i = j; i < m; ++i)
                        sum += tlapack::abs( A(i,j) );
                else {
                    sum = real_t( 1 );
                    for (idx_t i = j+1; i < m; ++i)
                        sum += tlapack::abs( A(i,j) );
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if ( isnan(sum) ) 
                        return sum;
                }
            }
        }
    }
    else
    {
        real_t scale(1), sum( 0 );

        if( uplo == Uplo::Upper ) {
            if( diag == Diag::NonUnit ) {
                for (idx_t j = 0; j < n; ++j)
                    lassq( slice(A,range<idx_t>(0,std::min(j+1,m)),j), scale, sum );
            }
            else {
                sum = real_t( std::min(m,n) );
                for (idx_t j = 1; j < n; ++j)
                    lassq( slice(A,range<idx_t>(0,std::min(j,m)),j), scale, sum );
            }
        }
        else {
            if( diag == Diag::NonUnit ) {
                for (idx_t j = 0; j < std::min(m,n); ++j)
                    lassq( slice(A,range<idx_t>(j,m),j), scale, sum );
            }
            else {
                sum = real_t( std::min(m,n) );
                for (idx_t j = 0; j < std::min(m-1,n); ++j)
                    lassq( slice(A,range<idx_t>(j+1,m),j), scale, sum );
            }
        }
        norm = scale * sqrt(sum);
    }

    return norm;
}

template<
    class norm_t, 
    class uplo_t, 
    class diag_t, 
    class matrix_t, 
    class work_t = undefined_t >
inline constexpr
void lantr_worksize(
    norm_t normType,
    uplo_t uplo,
    diag_t diag,
    const matrix_t& A,
    size_t& worksize,
    const workspace_opts_t<work_t>& opts )
{
    using T     = type_t< matrix_t >;
    using idx_t = size_type< matrix_t >;
    using vectorw_t = deduce_work_t< work_t, legacyVector<T,idx_t> >;

    worksize = sizeof( type_t< vectorw_t > ) * nrows(A);
}

/** Calculates the norm of a triangular matrix.
 * 
 * Code optimized for the infinity norm on column-major layouts using a workspace
 * of size at least m, where m is the number of rows of A.
 * @see lantr( norm_t normType, uplo_t uplo, diag_t diag, const matrix_t& A ).
 * 
 * @param work Vector of size at least m.
 * 
 * @ingroup auxiliary
 */
template<
    class norm_t, 
    class uplo_t, 
    class diag_t, 
    class matrix_t, 
    class work_t = undefined_t >
auto
lantr(
    norm_t normType,
    uplo_t uplo,
    diag_t diag,
    const matrix_t& A,
    size_t& worksize,
    const workspace_opts_t<work_t>& opts )
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
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper );
    tlapack_check_false( diag != Diag::NonUnit &&
                         diag != Diag::Unit );
    tlapack_check_false(  access_denied( uplo, read_policy(A) ) );

    // quick return
    if (m == 0 || n == 0)
        return real_t( 0 );

    // redirect for max-norm, one-norm and Frobenius norm
    if      ( normType == Norm::Max  ) return lantr( max_norm, uplo, diag, A );
    else if ( normType == Norm::One  ) return lantr( one_norm, uplo, diag,  A );
    else if ( normType == Norm::Fro  ) return lantr( frob_norm, uplo, diag, A );
    else if ( normType == Norm::Inf  ) {

        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
	    // the infinite norm.

        using vectorw_t = deduce_work_t< work_t, legacyVector<T,idx_t> >;

        // Allocates workspace
        vectorOfBytes localworkdata;
        const Workspace work = [&]()
        {
            size_t lwork;
            lantr_worksize( normType, uplo, diag, A, lwork, opts );
            return alloc_workspace( localworkdata, lwork, opts.work );
        }();
        auto w = Create< vectorw_t >( work, n, 1 );

        // Norm value
        real_t norm( 0 );

        if( uplo == Uplo::Upper ) {
            if( diag == Diag::NonUnit ) {
                for (idx_t i = 0; i < m; ++i)
                    w[i] = real_t(0);
    
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i <= std::min(j,m-1); ++i)
                        w[i] += tlapack::abs( A(i,j) );
            }
            else {
                for (idx_t i = 0; i < m; ++i)
                    w[i] = real_t(1);
    
                for (idx_t j = 1; j < n; ++j) {
                    for (idx_t i = 0; i < std::min(j,m); ++i)
                        w[i] += tlapack::abs( A(i,j) );
                }
            }
        }
        else {
            if( diag == Diag::NonUnit ) {
                for (idx_t i = 0; i < m; ++i)
                    w[i] = real_t(0);
    
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = j; i < m; ++i)
                        w[i] += tlapack::abs( A(i,j) );
            }
            else {
                for (idx_t i = 0; i < std::min(m,n); ++i)
                    w[i] = real_t(1);
                for (idx_t i = n; i < m; ++i)
                    w[i] = real_t(0);
    
                for (idx_t j = 1; j < n; ++j) {
                    for (idx_t i = j+1; i < m; ++i)
                        w[i] += tlapack::abs( A(i,j) );
                }
            }
        }

        for (idx_t i = 0; i < m; ++i)
        {
            real_t temp = w[i];

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

#endif // TLAPACK_LANTR_HH
