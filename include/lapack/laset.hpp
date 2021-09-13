/// @file laset.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/laset.h
//
// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LASET_HH__
#define __LASET_HH__

#include "lapack/types.hpp"

namespace lapack {

/** Initializes a matrix to diagonal and off-diagonal values
 *
 * @param[in] uplo Specifies whether the matrix A is upper or lower triangular:
 *
 *        'U': A is assumed to be upper triangular; elements below the diagonal are not referenced.
 *        'L': A is assumed to be lower triangular; elements above the diagonal are not referenced.
 *        otherwise, A is assumed to be a full matrix.
 * @param[in] m The number of rows of the matrix A.
 * @param[in] n The number of columns of the matrix A.
 * @param[in] alpha Value to assign to the off-diagonal elements of A.
 * @param[in] beta Value to assign to the diagonal elements of A.
 * @param[out] A Pointer to matrix A.
 * @param[in] lda Column length of the matrix A.
 * 
 * @ingroup auxiliary
 */
template< typename TA >
void laset(
    Uplo uplo, blas::idx_t m, blas::idx_t n,
    TA alpha, TA beta,
    TA* A, blas::idx_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    if (uplo == Uplo::Upper) {
        // Set the strictly upper triangular or trapezoidal part of
        // the array to alpha.
        for (idx_t j = 1; j < n; ++j) {
            const idx_t M = std::min(m,j);
            for (idx_t i = 0; i < M; ++i)
                A(i,j) = alpha;
        }
    }
    else if (uplo == Uplo::Lower) {
        // Set the strictly lower triangular or trapezoidal part of
        // the array to alpha.
        const idx_t N = std::min(m,n);
        for (idx_t j = 0; j < N; ++j) {
            for (idx_t i = j+1; i < m; ++i)
                A(i,j) = alpha;
        }
    }
    else {
        // Set the leading m-by-n submatrix to alpha.
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                A(i,j) = alpha;
    }

    // Set the first min(m,n) diagonal elements to beta.
    const idx_t N = std::min(m,n);
    for (idx_t i = 0; i < N; ++i)
        A(i,i) = beta;

    #undef A
}

/** Initializes a matrix to diagonal and off-diagonal values
 * 
 * @param[in] layout
 *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
 * @see laset( Uplo uplo, blas::idx_t m, blas::idx_t n, TA alpha, TA beta, TA* A, blas::idx_t lda )
 * 
 * @ingroup auxiliary
 */
template< typename TA >
inline void laset(
    Layout layout, Uplo uplo,
    blas::idx_t m, blas::idx_t n,
    TA alpha, TA beta,
    TA* A, blas::idx_t lda )
{
    if ( layout == Layout::RowMajor ) {
        if (uplo == Uplo::Upper) {
            // Set the Lower part instead of Upper
            uplo = Uplo::Lower;
        }
        else if (uplo == Uplo::Lower) {
            // Set the Upper part instead of Lower
            uplo = Uplo::Upper;
        }
        // Transpose A
        return laset(
	        uplo, n, m, alpha, beta, A, lda );
    }
    else {
        return laset(
	        uplo, m, n, alpha, beta, A, lda );
    }
}

}

#endif // __LASET_HH__
