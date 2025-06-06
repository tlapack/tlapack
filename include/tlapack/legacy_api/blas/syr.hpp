/// @file legacy_api/blas/syr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_SYR_HH
#define TLAPACK_LEGACY_SYR_HH

#include "tlapack/blas/syr.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * Symmetric matrix rank-1 update:
     * \[
     *     A = \alpha x x^T + A,
     * \]
     * where alpha is a scalar, x is a vector,
     * and A is an n-by-n symmetric matrix.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] layout
     *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
     *
     * @param[in] uplo
     *     What part of the matrix A is referenced,
     *     the opposite triangle being assumed from symmetry.
     *     - Uplo::Lower: only the lower triangular part of A is referenced.
     *     - Uplo::Upper: only the upper triangular part of A is referenced.
     *
     * @param[in] n
     *     Number of rows and columns of the matrix A. n >= 0.
     *
     * @param[in] alpha
     *     Scalar alpha. If alpha is zero, A is not updated.
     *
     * @param[in] x
     *     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
     *
     * @param[in] incx
     *     Stride between elements of x. incx must not be zero.
     *     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
     *
     * @param[in,out] A
     *     The n-by-n matrix A, stored in an lda-by-n array [RowMajor:
     * n-by-lda].
     *
     * @param[in] lda
     *     Leading dimension of A. lda >= max(1, n).
     *
     * @ingroup legacy_blas
     */
    template <typename TA, typename TX>
    void syr(Layout layout,
             Uplo uplo,
             idx_t n,
             scalar_type<TA, TX> alpha,
             TX const* x,
             int_t incx,
             TA* A,
             idx_t lda)
    {
        using internal::create_matrix;
        using scalar_t = scalar_type<TA, TX>;

        // check arguments
        tlapack_check_false(layout != Layout::ColMajor &&
                            layout != Layout::RowMajor);
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
        tlapack_check_false(n < 0);
        tlapack_check_false(incx == 0);
        tlapack_check_false(lda < n);

        // quick return
        if (n == 0 || alpha == scalar_t(0)) return;

        // for row major, swap lower <=> upper
        if (layout == Layout::RowMajor) {
            uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        }

        // Matrix views
        auto A_ = create_matrix<TA>(A, n, n, lda);

        tlapack_expr_with_vector(x_, TX, n, x, incx, syr(uplo, alpha, x_, A_));
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_SYR_HH
