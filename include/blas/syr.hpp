// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SYR_HH
#define TLAPACK_BLAS_SYR_HH

#include "base/utils.hpp"

namespace tlapack {

/**
 * Symmetric matrix rank-1 update:
 * \[
 *     A := \alpha x x^T + A,
 * \]
 * where alpha is a scalar, x is a vector,
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
 * @param[in,out] A A n-by-n symmetric matrix.
 *
 * @ingroup syr
 */
template< class matrixA_t, class vectorX_t, class alpha_t,
    class T = type_t<matrixA_t>,
    disable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >
    > = 0
>
void syr(
    Uplo uplo,
    const alpha_t& alpha,
    const vectorX_t& x,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t n = nrows(A);

    // check arguments
    tlapack_check_false( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    tlapack_check_false( size(x)  != n );
    tlapack_check_false( ncols(A) != n );

    tlapack_check_false( access_denied( uplo, write_policy(A) ) );

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = alpha * x[j];
            for (idx_t i = 0; i <= j; ++i)
                A(i,j) += x[i] * tmp;
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = alpha * x[j];
            for (idx_t i = j; i < n; ++i)
                A(i,j) += x[i] * tmp;
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_SYR_HH
