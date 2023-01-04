/// @file geru.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_GERU_HH
#define TLAPACK_BLAS_GERU_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/ger.hpp"

namespace tlapack {

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
 * @ingroup blas2
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
void geru(
    const alpha_t& alpha,
    const vectorX_t& x, const vectorY_t& y,
    matrixA_t& A )
{
    // data traits
    using idx_t = size_type< matrixA_t >;
    using scalar_t = scalar_type< alpha_t, type_t<vectorY_t> >;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false( size(x) != m );
    tlapack_check_false( size(y) != n );
    tlapack_check_false( access_denied( dense, write_policy(A) ) );

    for (idx_t j = 0; j < n; ++j) {
        const scalar_t tmp = alpha * y[j];
        for (idx_t i = 0; i < m; ++i)
            A(i,j) += x[i] * tmp;
    }
}

#ifdef USE_LAPACKPP_WRAPPERS

    template<
        class matrixA_t,
        class vectorX_t, class vectorY_t,
        class alpha_t,
        class T = type_t<matrixA_t>,
        enable_if_allow_optblas_t<
            pair< alpha_t, T >,
            pair< matrixA_t, T >,
            pair< vectorX_t, T >,
            pair< vectorY_t, T >
        > = 0
    >
    inline
    void geru(
        const alpha_t alpha,
        const vectorX_t& x, const vectorY_t& y,
        matrixA_t& A )
    {
        using idx_t = size_type< matrixA_t >;

        // Legacy objects
        auto A_ = legacy_matrix(A);
        auto x_ = legacy_vector(x);
        auto y_ = legacy_vector(y);

        // Constants to forward
        const idx_t& m = A_.m;
        const idx_t& n = A_.n;
        const idx_t incx = (x_.direction == Direction::Forward) ? idx_t(x_.inc) : idx_t(-x_.inc);
        const idx_t incy = (y_.direction == Direction::Forward) ? idx_t(y_.inc) : idx_t(-y_.inc);

        return ::blas::geru(
            (::blas::Layout) A_.layout,
            m, n,
            alpha,
            x_.ptr, incx,
            y_.ptr, incy,
            A_.ptr, A_.ldim );
    }

#endif

}  // namespace tlapack

#endif        //  #ifndef TLAPACK_BLAS_GER_HH
