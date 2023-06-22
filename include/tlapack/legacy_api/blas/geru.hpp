/// @file legacy_api/blas/geru.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_GERU_HH
#define TLAPACK_LEGACY_GERU_HH

#include "tlapack/blas/geru.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * General matrix rank-1 update:
     * \[
     *     A = \alpha x y^T + A,
     * \]
     * where alpha is a scalar, x and y are vectors,
     * and A is an m-by-n matrix.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] layout
     *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
     *
     * @param[in] m
     *     Number of rows of the matrix A. m >= 0.
     *
     * @param[in] n
     *     Number of columns of the matrix A. n >= 0.
     *
     * @param[in] alpha
     *     Scalar alpha. If alpha is zero, A is not updated.
     *
     * @param[in] x
     *     The m-element vector x, in an array of length (m-1)*abs(incx) + 1.
     *
     * @param[in] incx
     *     Stride between elements of x. incx must not be zero.
     *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
     *
     * @param[in] y
     *     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
     *
     * @param[in] incy
     *     Stride between elements of y. incy must not be zero.
     *     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
     *
     * @param[in,out] A
     *     The m-by-n matrix A, stored in an lda-by-n array [RowMajor:
     * m-by-lda].
     *
     * @param[in] lda
     *     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1,
     * n)].
     *
     * @ingroup legacy_blas
     */
    template <typename TA, typename TX, typename TY>
    void geru(Layout layout,
              idx_t m,
              idx_t n,
              scalar_type<TA, TX, TY> alpha,
              TX const* x,
              int_t incx,
              TY const* y,
              int_t incy,
              TA* A,
              idx_t lda)
    {
        using internal::create_matrix;
        using scalar_t = scalar_type<TA, TX, TY>;

        // check arguments
        tlapack_check_false(layout != Layout::ColMajor &&
                            layout != Layout::RowMajor);
        tlapack_check_false(m < 0);
        tlapack_check_false(n < 0);
        tlapack_check_false(incx == 0);
        tlapack_check_false(incy == 0);
        tlapack_check_false(lda < ((layout == Layout::ColMajor) ? m : n));

        // quick return
        if (m == 0 || n == 0 || alpha == scalar_t(0)) return;

        if (layout == Layout::ColMajor) {
            // Matrix views
            auto A_ = create_matrix<TA>(A, m, n, lda);

            tlapack_expr_with_2vectors(x_, TX, m, x, incx, y_, TY, n, y, incy,
                                       return geru(alpha, x_, y_, A_));
        }
        else {
            // Matrix views
            auto A_ = create_matrix<TA>(A, n, m, lda);

            tlapack_expr_with_2vectors(y_, TY, n, y, incy, x_, TX, m, x, incx,
                                       return geru(alpha, y_, x_, A_));
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_GER_HH
