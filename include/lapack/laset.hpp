// Copyright (c) 2012-2021, University of Colorado Denver. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// Created by
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Adapted from https://github.com/langou/latl/blob/master/include/laset.h
/// @author Rodney James, University of Colorado Denver, USA

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
    Uplo uplo, blas::size_t m, blas::size_t n,
    TA alpha, TA beta,
    TA* A, blas::size_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    if (uplo == Uplo::Upper) {
        // Set the strictly upper triangular or trapezoidal part of
        // the array to alpha.
        for (size_t j = 1; j < n; ++j)
            for (size_t i = 0; i < std::min(m,j); ++i)
                A(i,j) = alpha;
    }
    else if (uplo == Uplo::Lower) {
        // Set the strictly lower triangular or trapezoidal part of
        // the array to alpha.
        for (size_t j = 0; j < std::min(m,n); ++j)
            for (size_t i = j+1; i < m; ++i)
                A(i,j) = alpha;
    }
    else {
        // Set the leading m-by-n submatrix to alpha.
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                A(i,j) = alpha;
    }

    // Set the first min(M,N) diagonal elements to beta.
    for (size_t i = 0; i < std::min(m,n); ++i)
        A(i,i) = beta;

    #undef A
}

}

#endif // __LASET_HH__