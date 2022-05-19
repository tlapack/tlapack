/// @file lacpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/lacpy.h
//
// Copyright (c) 2012-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LACPY_HH__
#define __TLAPACK_LACPY_HH__

#include "base/types.hpp"

namespace tlapack {

/**
 * @brief Copies a matrix from A to B.
 * 
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * 
 * @param[in] uplo
 *      - Uplo::Upper:   Upper triangle of A and B are referenced;
 *      - Uplo::Lower:   Lower triangle of A and B are referenced;
 *      - Uplo::General: All entries of A are referenced; the first m rows of B
 *                          and first n columns of B are referenced.
 * 
 * @param A m-by-n matrix.
 * @param B matrix with at least m rows and at least n columns.
 * 
 * @ingroup auxiliary
 */
template< class uplo_t, class matrixA_t, class matrixB_t >
void lacpy( uplo_t uplo, const matrixA_t& A, matrixB_t& B )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper &&
                    uplo != Uplo::General );
    tlapack_check_false(  access_denied( uplo, read_policy(A) ) );
    tlapack_check_false(  access_denied( uplo, write_policy(B) ) );

    if( uplo == Uplo::Upper ) {
        // Set the strictly upper triangular or trapezoidal part of B
        for (idx_t j = 0; j < n; ++j) {
            const auto M = std::min( m, j+1 );
            for (idx_t i = 0; i < M; ++i)
                B(i,j) = A(i,j);
        }
    }
    else if( uplo == Uplo::Lower ) {
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
