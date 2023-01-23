/// @file axpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_AXPY_HH
#define TLAPACK_LEGACY_AXPY_HH

#include "tlapack/blas/axpy.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {

/**
 * Add scaled vector, $y = \alpha x + y$.
 *
 * Wrapper to axpy(
    const alpha_t& alpha,
    const vectorX_t& x, vectorY_t& y ).
 *
 * @param[in] n
 *     Number of elements in x and y. n >= 0.
 *
 * @param[in] alpha
 *     Scalar alpha. If alpha is zero, y is not updated.
 *
 * @param[in] x
 *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 *
 * @param[in] incx
 *     Stride between elements of x. incx must not be zero.
 *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 *
 * @param[in,out] y
 *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 *
 * @param[in] incy
 *     Stride between elements of y. incy must not be zero.
 *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 *
 * @ingroup legacy_blas
 */
template <typename TX, typename TY>
void axpy(idx_t n,
          scalar_type<TX, TY> alpha,
          TX const* x,
          int_t incx,
          TY* y,
          int_t incy)
{
    tlapack_check_false(incx == 0);
    tlapack_check_false(incy == 0);
    using scalar_t = scalar_type<TX, TY>;

    // quick return
    if (n <= 0 || alpha == scalar_t(0)) return;

    tlapack_expr_with_2vectors(x_, TX, n, x, incx, y_, TY, n, y, incy,
                               return axpy(alpha, x_, y_));
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_AXPY_HH
