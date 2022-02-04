// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GER_HH
#define BLAS_GER_HH

#include "blas/utils.hpp"

namespace blas {

/**
 * General matrix rank-1 update:
 * \[
 *     A = \alpha x y^H + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an m-by-n matrix.
 *
 * Generic implementation for arbitrary data types.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, A is not updated.
 *
 * @param[in] x
 *     The m-element vector x, in an array of length (m-1)*abs(incx) + 1.
 *
 * @param[in] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in, out] A
 *     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
 *
 * @ingroup ger
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t >
void ger(
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t j = 0; j < n; ++j) {
        auto tmp = alpha * conj( y[j] );
        for (idx_t i = 0; i < m; ++i)
            A(i,j) += x[i] * tmp;
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_GER_HH
