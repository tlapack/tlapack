// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GERU_HH
#define BLAS_GERU_HH

#include "blas/utils.hpp"
#include "blas/ger.hpp"

namespace blas {

/**
 * General matrix rank-1 update:
 * \[
 *     A := \alpha x y^T + A,
 * \]
 * where alpha is a scalar, x and y are vectors,
 * and A is an m-by-n matrix.
 * 
 * @param[in] alpha Scalar.
 * @param[in] x A m-element vector.
 * @param[in] y A n-element vector.
 * @param[in] A A m-by-n matrix.
 * 
 * @ingroup geru
 */
template<
    class matrixA_t,
    class vectorX_t, class vectorY_t,
    class alpha_t >
void geru(
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    blas_error_if( size(x) != m );
    blas_error_if( size(y) != n );
    blas_error_if( access_denied( dense, write_policy(A) ) );

    for (idx_t j = 0; j < n; ++j) {
        auto tmp = alpha * y[j];
        for (idx_t i = 0; i < m; ++i)
            A(i,j) += x[i] * tmp;
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_GER_HH
