/// @file legacy_api/blas/symm.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_SYMM_HH
#define TLAPACK_LEGACY_SYMM_HH

#include "tlapack/blas/symm.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * Symmetric matrix-matrix multiply:
     * \[
     *     C = \alpha A B + \beta C,
     * \]
     * or
     * \[
     *     C = \alpha B A + \beta C,
     * \]
     * where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric
     * matrix, and B and C are m-by-n matrices.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] layout
     *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
     *
     * @param[in] side
     *     The side the matrix A appears on:
     *     - Side::Left:  $C = \alpha A B + \beta C$,
     *     - Side::Right: $C = \alpha B A + \beta C$.
     *
     * @param[in] uplo
     *     What part of the matrix A is referenced:
     *     - Uplo::Lower: only the lower triangular part of A is referenced.
     *     - Uplo::Upper: only the upper triangular part of A is referenced.
     *
     * @param[in] m
     *     Number of rows of the matrices B and C.
     *
     * @param[in] n
     *     Number of columns of the matrices B and C.
     *
     * @param[in] alpha
     *     Scalar alpha. If alpha is zero, A and B are not accessed.
     *
     * @param[in] A
     *     - If side = Left:  The m-by-m matrix A, stored in an lda-by-m array.
     *     - If side = Right: The n-by-n matrix A, stored in an lda-by-n array.
     *
     * @param[in] lda
     *     Leading dimension of A.
     *     - If side = Left:  lda >= max(1, m).
     *     - If side = Right: lda >= max(1, n).
     *
     * @param[in] B
     *     The m-by-n matrix B, stored in an ldb-by-n array.
     *
     * @param[in] ldb
     *     Leading dimension of B. ldb >= max(1, n).
     *
     * @param[in] beta
     *     Scalar beta. If beta is zero, C need not be set on input.
     *
     * @param[in] C
     *     The m-by-n matrix C, stored in an lda-by-n array.
     *
     * @param[in] ldc
     *     Leading dimension of C. ldc >= max(1, n).
     *
     * @ingroup legacy_blas
     */
    template <typename TA, typename TB, typename TC>
    void symm(Layout layout,
              Side side,
              Uplo uplo,
              idx_t m,
              idx_t n,
              scalar_type<TA, TB, TC> alpha,
              TA const* A,
              idx_t lda,
              TB const* B,
              idx_t ldb,
              scalar_type<TA, TB, TC> beta,
              TC* C,
              idx_t ldc)
    {
        using scalar_t = scalar_type<TA, TB, TC>;

        // check arguments
        tlapack_check_false(layout != Layout::ColMajor &&
                            layout != Layout::RowMajor);
        tlapack_check_false(side != Side::Left && side != Side::Right);
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                            uplo != Uplo::General);
        tlapack_check_false(m < 0);
        tlapack_check_false(n < 0);
        tlapack_check_false(lda < ((side == Side::Left) ? m : n));
        tlapack_check_false(ldb < ((layout == Layout::RowMajor) ? n : m));
        tlapack_check_false(ldc < ((layout == Layout::RowMajor) ? n : m));

        // quick return
        if (m == 0 || n == 0 ||
            ((alpha == scalar_t(0)) && (beta == scalar_t(1))))
            return;

        // adapt if row major
        if (layout == Layout::RowMajor) {
            side = (side == Side::Left) ? Side::Right : Side::Left;
            if (uplo == Uplo::Lower)
                uplo = Uplo::Upper;
            else if (uplo == Uplo::Upper)
                uplo = Uplo::Lower;
            std::swap(m, n);
        }

        // Matrix views
        const auto A_ = (side == Side::Left)
                            ? create_matrix<TA>((TA*)A, m, m, lda)
                            : create_matrix<TA>((TA*)A, n, n, lda);
        const auto B_ = create_matrix<TB>((TB*)B, m, n, ldb);
        auto C_ = create_matrix<TC>(C, m, n, ldc);

        if (alpha == scalar_t(0)) {
            if (beta == scalar_t(0)) {
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C_(i, j) = TC(0);
            }
            else {
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < m; ++i)
                        C_(i, j) *= beta;
            }
        }
        else {
            if (beta == scalar_t(0))
                symm(side, uplo, alpha, A_, B_, C_);
            else
                symm(side, uplo, alpha, A_, B_, beta, C_);
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_SYMM_HH
