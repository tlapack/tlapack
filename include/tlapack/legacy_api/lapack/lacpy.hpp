/// @file legacy_api/lapack/lacpy.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lacpy.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LACPY_HH
#define TLAPACK_LEGACY_LACPY_HH

#include "tlapack/lapack/lacpy.hpp"

namespace tlapack {
namespace legacy {

    /**
     * @brief Copies a matrix from A to B.
     *
     * @tparam uplo_t
     *      Either Uplo or any class that implements `operator Uplo()`.
     *
     * @param[in] uplo
     *      - Uplo::Upper:   Upper triangle of A and B are referenced;
     *      - Uplo::Lower:   Lower triangle of A and B are referenced;
     *      - Uplo::General: All entries of A are referenced; the first m rows
     * of B and first n columns of B are referenced.
     * @param[in] m Number of rows of A.
     * @param[in] n Number of columns of A.
     * @param[in] A m-by-n matrix.
     * @param[in] lda Leading dimension of A.
     * @param[out] B Matrix with at least m rows and at least n columns.
     * @param[in] ldb Leading dimension of B.
     */
    template <class uplo_t, typename TA, typename TB>
    void lacpy(
        uplo_t uplo, idx_t m, idx_t n, const TA* A, idx_t lda, TB* B, idx_t ldb)
    {
        using internal::create_matrix;

        // check arguments
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                            uplo != Uplo::General);

        // Matrix views
        const auto A_ = create_matrix<TA>((TA*)A, m, n, lda);
        auto B_ = create_matrix<TB>(B, m, n, ldb);

        lacpy(uplo, A_, B_);
    }

    /** Copies a real matrix from A to B where A is either a full, upper
     * triangular or lower triangular matrix.
     *
     * @param[in] matrixtype :
     *
     *        'U': A is assumed to be upper triangular; elements below the
     * diagonal are not referenced. 'L': A is assumed to be lower triangular;
     * elements above the diagonal are not referenced. otherwise, A is assumed
     * to be a full matrix.
     *
     * @param[in] m Number of rows of A.
     * @param[in] n Number of columns of A.
     * @param[in] A m-by-n matrix.
     * @param[in] lda Leading dimension of A.
     * @param[out] B Matrix with at least m rows and at least n columns.
     * @param[in] ldb Leading dimension of B.
     *
     * @see lacpy( uplo_t, idx_t, idx_t, const TA*, idx_t, TB* B, idx_t )
     *
     * @ingroup legacy_lapack
     */
    template <typename TA, typename TB>
    void lacpy(MatrixType matrixtype,
               idx_t m,
               idx_t n,
               const TA* A,
               idx_t lda,
               TB* B,
               idx_t ldb)
    {
        if (matrixtype == MatrixType::Upper) {
            lacpy(UPPER_TRIANGLE, m, n, A, lda, B, ldb);
        }
        else if (matrixtype == MatrixType::Lower) {
            lacpy(LOWER_TRIANGLE, m, n, A, lda, B, ldb);
        }
        else {
            lacpy(GENERAL, m, n, A, lda, B, ldb);
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LACPY_HH
