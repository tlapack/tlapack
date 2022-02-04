/// @file lacpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lacpy.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LACPY_HH__
#define __LACPY_HH__

#include "lapack/types.hpp"

namespace lapack {

/** Copies a real matrix from A to B where A is either a full, upper triangular or lower triangular matrix.
 *
 * @param[in] uplo Specifies whether the matrix A is upper or lower triangular:
 *
 *        'U': A is assumed to be upper triangular; elements below the diagonal are not referenced.
 *        'L': A is assumed to be lower triangular; elements above the diagonal are not referenced.
 *        otherwise, A is assumed to be a full matrix.
 * @param[in] m The number of rows of the matrix A.
 * @param[in] n The number of columns of the matrix A.
 * @param[out] A Pointer to matrix A.
 * @param[in] lda Column length of the matrix A.
 * @param[out] B Pointer to matrix B.
 * @param[in] ldb Column length of the matrix B.
 * 
 * @ingroup auxiliary
 */
template< class uplo_t, class matrixA_t, class matrixB_t,
    enable_if_t<(
    /* Requires: */
        is_same_v< uplo_t, upper_triangle_t > || 
        is_same_v< uplo_t, lower_triangle_t > || 
        is_same_v< uplo_t, general_matrix_t >
    ), int > = 0
>
void lacpy( uplo_t uplo, const matrixA_t& A, matrixB_t& B )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const auto m = nrows(A);
    const auto n = ncols(A);

    if( is_same_v< uplo_t, upper_triangle_t > ) {
        // Set the strictly upper triangular or trapezoidal part of B
        for (idx_t j = 0; j < n; ++j) {
            const auto M = std::min( m, j+1 );
            for (idx_t i = 0; i < M; ++i)
                B(i,j) = A(i,j);
        }
    }
    else if( is_same_v< uplo_t, lower_triangle_t > ) {
        // Set the strictly lower triangular or trapezoidal part of B
        const auto N = std::min(m,n);
        for (idx_t j = 0; j < N; ++j)
            for (idx_t i = j; i < m; ++i)
                B(i,j) = A(i,j);
    }
    else {
        // Set the whole m-by-n matrix B
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                B(i,j) = A(i,j);
    }
}

}

#endif // __LACPY_HH__
