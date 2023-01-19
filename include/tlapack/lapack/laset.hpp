/// @file laset.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/laset.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LASET_HH
#define TLAPACK_LASET_HH

#include "tlapack/base/types.hpp"

namespace tlapack {

/**
 * @brief Initializes a matrix to diagonal and off-diagonal values.
 * 
 * @tparam uplo_t
 *      Either Uplo or any class that implements `operator Uplo()`.
 * 
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced;
 *      - Uplo::General: All entries of A are referenced.
 *
 * @param[in] alpha Value to assign to the off-diagonal elements of A.
 * @param[in] beta Value to assign to the diagonal elements of A.
 * 
 * @param[out] A m-by-n matrix.
 * 
 * @ingroup auxiliary 
 */
template< class uplo_t, class matrix_t >
void laset(
    uplo_t uplo,
    const type_t<matrix_t>& alpha, const type_t<matrix_t>& beta,
    matrix_t& A )
{
    using idx_t  = size_type< matrix_t >;
    using std::min;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(  uplo != Uplo::Lower &&
                    uplo != Uplo::Upper &&
                    uplo != Uplo::General );
    tlapack_check_false(  access_denied( uplo, write_policy(A) ) );

    if (uplo == Uplo::Upper) {
        // Set the strictly upper triangular or trapezoidal part of
        // the array to alpha.
        for (idx_t j = 1; j < n; ++j) {
            const idx_t M = min(m,j);
            for (idx_t i = 0; i < M; ++i)
                A(i,j) = alpha;
        }
    }
    else if (uplo == Uplo::Lower) {
        // Set the strictly lower triangular or trapezoidal part of
        // the array to alpha.
        const idx_t N = min(m,n);
        for (idx_t j = 0; j < N; ++j) {
            for (idx_t i = j+1; i < m; ++i)
                A(i,j) = alpha;
        }
    }
    else {
        // Set all elements in A to alpha.
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i,j) = alpha;
    }

    // Set the first min(m,n) diagonal elements to beta.
    const idx_t N = min(m,n);
    for (idx_t i = 0; i < N; ++i)
        A(i,i) = beta;
}

}

#endif // TLAPACK_LASET_HH
