// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYR2_HH
#define BLAS_SYR2_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * Symmetric matrix rank-2 update:
 * \[
 *     A := \alpha x y^T + \alpha y x^T + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an n-by-n symmetric matrix.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 * 
 * @param[in] alpha Scalar.
 * @param[in] x A n-element vector.
 * @param[in] y A n-element vector.
 * @param[in,out] A A n-by-n symmetric matrix.
 *
 * @ingroup syr2
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t >
void syr2(
    blas::Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( size(x)  != n );
    blas_error_if( size(y)  != n );
    blas_error_if( ncols(A) != n );

    blas_error_if( access_denied( uplo, write_policy(A) ) );

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * y[j];
            auto tmp2 = alpha * x[j];
            for (idx_t i = 0; i <= j; ++i)
                A(i,j) += x[i]*tmp1 + y[i]*tmp2;
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * y[j];
            auto tmp2 = alpha * x[j];
            for (idx_t i = j; i < n; ++i)
                A(i,j) += x[i]*tmp1 + y[i]*tmp2;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYR2_HH
