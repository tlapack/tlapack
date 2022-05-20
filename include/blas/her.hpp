// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_BLAS_HER_HH__
#define __TLAPACK_BLAS_HER_HH__

#include "base/utils.hpp"

namespace tlapack {

/**
 * Hermitian matrix rank-1 update:
 * \[
 *     A := \alpha x x^H + A,
 * \]
 * where alpha is a real scalar, x is a vector,
 * and A is an n-by-n Hermitian matrix.
 * 
 * Mind that if alpha is complex, the output matrix is no longer Hermitian.
 *
 * @param[in] uplo
 *     What part of the matrix A is referenced,
 *     the opposite triangle being assumed from symmetry.
 *     - Uplo::Lower: only the lower triangular part of A is referenced.
 *     - Uplo::Upper: only the upper triangular part of A is referenced.
 *
 * @param[in] alpha Real scalar.
 * @param[in] x A n-element vector.
 * @param[in,out] A A n-by-n Hermitian matrix.
 *     Imaginary parts of the diagonal elements need not be set,
 *     are assumed to be zero on entry, and are set to zero on exit.
 *
 * @ingroup her
 */
template< class matrixA_t, class vectorX_t, class alpha_t,
    enable_if_t<(
    /* Requires: */
        ! is_complex<alpha_t>::value
    ), int > = 0,
    disable_if_allow_optblas_t<
        pair< alpha_t, real_type<type_t<matrixA_t>> >,
        pair< matrixA_t, type_t<matrixA_t> >,
        pair< vectorX_t, type_t<matrixA_t> >
    > = 0
>
void her(
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
            auto tmp = alpha * conj( x[j] );
            for (idx_t i = 0; i < j; ++i)
                A(i,j) += x[i] * tmp;
            A(j,j) = real( A(j,j) ) + real( x[j] * tmp );
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            auto tmp = alpha * conj( x[j] );
            A(j,j) = real( A(j,j) ) + real( tmp * x[j] );
            for (idx_t i = j+1; i < n; ++i)
                A(i,j) += x[i] * tmp;
        }
    }
}

}  // namespace tlapack

#endif        //  #ifndef __TLAPACK_BLAS_HER_HH__
