/// @file legacy_api/blas/syrk.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_SYRK_HH
#define TLAPACK_LEGACY_SYRK_HH

#include "tlapack/blas/syrk.hpp"
#include "tlapack/legacy_api/base/types.hpp"
#include "tlapack/legacy_api/base/utils.hpp"

namespace tlapack {
namespace legacy {

    /**
     * Symmetric rank-k update:
     * \[
     *     C = \alpha A A^T + \beta C,
     * \]
     * or
     * \[
     *     C = \alpha A^T A + \beta C,
     * \]
     * where alpha and beta are scalars, C is an n-by-n symmetric matrix,
     * and A is an n-by-k or k-by-n matrix.
     *
     * Generic implementation for arbitrary data types.
     *
     * @param[in] layout
     *     Matrix storage, Layout::ColMajor or Layout::RowMajor.
     *
     * @param[in] uplo
     *     What part of the matrix C is referenced,
     *     the opposite triangle being assumed from symmetry:
     *     - Uplo::Lower: only the lower triangular part of C is referenced.
     *     - Uplo::Upper: only the upper triangular part of C is referenced.
     *
     * @param[in] trans
     *     The operation to be performed:
     *     - Op::NoTrans: $C = \alpha A A^T + \beta C$.
     *     - Op::Trans:   $C = \alpha A^T A + \beta C$.
     *     - In the real    case, Op::ConjTrans is interpreted as Op::Trans.
     *       In the complex case, Op::ConjTrans is illegal (see herk()
     * instead).
     *
     * @param[in] n
     *     Number of rows and columns of the matrix C. n >= 0.
     *
     * @param[in] k
     *     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
     *     - Otherwise:          number of rows    of the matrix A. k >= 0.
     *
     * @param[in] alpha
     *     Scalar alpha. If alpha is zero, A is not accessed.
     *
     * @param[in] A
     *     - If trans = NoTrans:
     *       the n-by-k matrix A, stored in an lda-by-k array [RowMajor:
     * n-by-lda].
     *     - Otherwise:
     *       the k-by-n matrix A, stored in an lda-by-n array [RowMajor:
     * k-by-lda].
     *
     * @param[in] lda
     *     Leading dimension of A.
     *     - If trans = NoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)],
     *     - Otherwise:              lda >= max(1, k) [RowMajor: lda >= max(1,
     * n)].
     *
     * @param[in] beta
     *     Scalar beta. If beta is zero, C need not be set on input.
     *
     * @param[in] C
     *     The n-by-n symmetric matrix C,
     *     stored in an lda-by-n array [RowMajor: n-by-lda].
     *
     * @param[in] ldc
     *     Leading dimension of C. ldc >= max(1, n).
     *
     * @ingroup legacy_blas
     */
    template <typename TA, typename TC>
    void syrk(Layout layout,
              Uplo uplo,
              Op trans,
              idx_t n,
              idx_t k,
              scalar_type<TA, TC> alpha,
              TA const* A,
              idx_t lda,
              scalar_type<TA, TC> beta,
              TC* C,
              idx_t ldc)
    {
        using internal::create_matrix;
        using scalar_t = scalar_type<TA, TC>;

        // check arguments
        tlapack_check_false(layout != Layout::ColMajor &&
                            layout != Layout::RowMajor);
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                            uplo != Uplo::General);
        tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                            trans != Op::ConjTrans);
        tlapack_check_false(is_complex<TA> && trans == Op::ConjTrans);
        tlapack_check_false(n < 0);
        tlapack_check_false(k < 0);
        tlapack_check_false(lda < ((layout == Layout::RowMajor)
                                       ? ((trans == Op::NoTrans) ? k : n)
                                       : ((trans == Op::NoTrans) ? n : k)));
        tlapack_check_false(ldc < n);

        // quick return
        if (n == 0 ||
            ((alpha == scalar_t(0) || k == 0) && (beta == scalar_t(1))))
            return;

        // This algorithm only works with Op::NoTrans or Op::Trans
        if (trans == Op::ConjTrans) trans = Op::Trans;

        // adapt if row major
        if (layout == Layout::RowMajor) {
            if (uplo == Uplo::Lower)
                uplo = Uplo::Upper;
            else if (uplo == Uplo::Upper)
                uplo = Uplo::Lower;
            trans = (trans == Op::NoTrans) ? Op::Trans : Op::NoTrans;
        }

        // Matrix views
        const auto A_ = (trans == Op::NoTrans)
                            ? create_matrix<TA>((TA*)A, n, k, lda)
                            : create_matrix<TA>((TA*)A, k, n, lda);
        auto C_ = create_matrix<TC>(C, n, n, ldc);

        if (alpha == scalar_t(0)) {
            if (beta == scalar_t(0)) {
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < n; ++i)
                        C_(i, j) = TC(0);
            }
            else {
                for (idx_t j = 0; j < n; ++j)
                    for (idx_t i = 0; i < n; ++i)
                        C_(i, j) *= beta;
            }
        }
        else {
            if (beta == scalar_t(0))
                syrk(uplo, trans, alpha, A_, C_);
            else
                syrk(uplo, trans, alpha, A_, beta, C_);
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  //  #ifndef TLAPACK_LEGACY_SYMM_HH
