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
template< typename TA, typename TB >
void lacpy(
    Uplo uplo, blas::idx_t m, blas::idx_t n,
    const TA* A, blas::idx_t lda,
    TB* B, blas::idx_t ldb )
{
    using blas::view_matrix;
    
    // Matrix views
    auto _A = view_matrix<const TA>( A, m, n, lda );
    auto _B = view_matrix<TB>( B, m, n, ldb );

    if (uplo == Uplo::Upper) {
        // Set the strictly upper triangular or trapezoidal part of B
        for (idx_t j = 0; j < n; ++j) {
            const idx_t M = std::min<idx_t>( m, j+1 );
            for (idx_t i = 0; i < M; ++i)
                _B(i,j) = _A(i,j);
        }
    }
    else if (uplo == Uplo::Lower) {
        // Set the strictly lower triangular or trapezoidal part of B
        const idx_t N = std::min(m,n);
        for (idx_t j = 0; j < N; ++j)
            for (idx_t i = j; i < m; ++i)
                _B(i,j) = _A(i,j);
    }
    else {
        // Set the whole m-by-n matrix B
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                _B(i,j) = _A(i,j);
    }
}

/** Copies a real matrix from A to B where A is either a full, upper triangular or lower triangular matrix.
 *
 * @param[in] matrixtype :
 *
 *        'U': A is assumed to be upper triangular; elements below the diagonal are not referenced.
 *        'L': A is assumed to be lower triangular; elements above the diagonal are not referenced.
 *        otherwise, A is assumed to be a full matrix.
 * 
 * @see lacpy( Uplo, blas::idx_t, blas::idx_t, TA*, blas::idx_t, TB* B, blas::idx_t )
 * 
 * @ingroup auxiliary
 */
template< typename TA, typename TB >
void inline lacpy(
    MatrixType matrixtype, blas::idx_t m, blas::idx_t n,
    TA* A, blas::idx_t lda,
    TB* B, blas::idx_t ldb )
{
    if (matrixtype == MatrixType::Upper) {
        lacpy(Uplo::Upper, m, n, A, lda, B, ldb);
    } else if (matrixtype == MatrixType::Lower) {
        lacpy(Uplo::Lower, m, n, A, lda, B, ldb);
    } else {
        lacpy(Uplo::General, m, n, A, lda, B, ldb);
    }
}

}

#endif // __LACPY_HH__
