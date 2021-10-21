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
inline real_type<TA> lange(
    Norm normType, blas::idx_t m, blas::idx_t n,
    const TA *_A, blas::idx_t lda )
{
    using blas::internal::colmajor_matrix;
    const auto A = colmajor_matrix<TA>( (TA*)_A, m, n, lda );

    if( normType == Norm::Max )
        return lange( max_norm, A );
    else if ( normType == Norm::One )
        return lange( one_norm, A );
    else if ( normType == Norm::Inf )
        return lange( inf_norm, A );
    else if ( normType == Norm::Fro )
        return lange( frob_norm, A );
    else
        return real_type<TA>( 0.0 );
}

} // lapack

#endif // __LANGE_HH__
