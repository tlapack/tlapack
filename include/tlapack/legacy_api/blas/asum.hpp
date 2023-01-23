/// @file asum.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_ASUM_HH
#define TLAPACK_LEGACY_ASUM_HH

#include "tlapack/blas/asum.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {

/**
 * Wrapper to asum( vector_t const& x ).
 *
 * @return 1-norm of vector,
 *     $|| Re(x) ||_1 + || Im(x) ||_1
 *         = \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
 *
 * @param[in] n
 *     Number of elements in x. n >= 0.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*incx + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx > 0.
 *
 * @ingroup legacy_blas
 */
template <typename T>
inline real_type<T> asum(idx_t n, T const* x, int_t incx)
{
    tlapack_check_false(incx <= 0);

    // quick return
    if (n <= 0) return 0;

    tlapack_expr_with_vector_positiveInc(x_, T, n, x, incx, return asum(x_));
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_ASUM_HH
