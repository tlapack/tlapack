// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_HER2_HH
#define TLAPACK_BLAS_HER2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix rank-2 update:
 * \[
 *     A := \alpha x y^H + \text{conj}(\alpha) y x^H + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an n-by-n Hermitian matrix.
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
 * @param[in,out] A A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @ingroup her2
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t,
    class T = type_t<matrixA_t>,
    disable_if_allow_optblas_t<
        pair< alpha_t, T >,
        pair< matrixA_t, T >,
        pair< vectorX_t, T >,
        pair< vectorY_t, T >
    > = 0
>
void her2(
    Uplo  uplo,
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
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
    tlapack_check_false( size(y)  != n );
    tlapack_check_false( ncols(A) != n );

    tlapack_check_false( access_denied( uplo, write_policy(A) ) );

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * conj( y[j] );
            auto tmp2 = conj( alpha * x[j] );
            for (idx_t i = 0; i < j; ++i)
                A(i,j) += x[i]*tmp1 + y[i]*tmp2;
            A(j,j) = real( A(j,j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp1 = alpha * conj( y[j] );
            auto tmp2 = conj( alpha * x[j] );
            A(j,j) = real( A(j,j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
            for (idx_t i = j+1; i < n; ++i)
                A(i,j) += x[i]*tmp1 + y[i]*tmp2;
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_HER2_HH
