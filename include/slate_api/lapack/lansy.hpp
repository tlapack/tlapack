/// @file lansy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SLATE_LANSY_HH__
#define __SLATE_LANSY_HH__

#include "lapack/types.hpp"
#include "lapack/lansy.hpp"

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
template <typename TA>
real_type<TA> lansy(
    Norm normType, Uplo uplo, blas::idx_t n,
    const TA *A, blas::idx_t lda )
{
    typedef real_type<TA> real_t;
    using blas::internal::colmajor_matrix;

    // constants
    const real_t zero( 0 );

    // quick return
    if ( n == 0 ) return zero;

    // Matrix views
    const auto _A = colmajor_matrix<TA>( (TA*)A, n, n, lda );

    if( normType == Norm::Max ) {
        if( uplo == Uplo::Upper )   return lansy( max_norm, upper_triangle, _A );
        else                        return lansy( max_norm, lower_triangle, _A );
    }
    else if ( normType == Norm::One ||normType == Norm::Inf ) {
        if( uplo == Uplo::Upper )   return lansy( one_norm, upper_triangle, _A );
        else                        return lansy( one_norm, lower_triangle, _A );
    }
    else if ( normType == Norm::Fro ) {
        if( uplo == Uplo::Upper )   return lansy( frob_norm, upper_triangle, _A );
        else                        return lansy( frob_norm, lower_triangle, _A );
    }

    return zero;
}

} // lapack

#endif // __SLATE_LANSY_HH__
