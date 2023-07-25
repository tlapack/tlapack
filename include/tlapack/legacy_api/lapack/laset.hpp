/// @file legacy_api/lapack/laset.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/laset.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LASET_HH
#define TLAPACK_LEGACY_LASET_HH

#include "tlapack/lapack/laset.hpp"

namespace tlapack {
namespace legacy {

    /**
     * @brief Initializes a matrix to diagonal and off-diagonal values
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
     * @param[in] alpha Value to be assigned to the off-diagonal elements of A.
     * @param[in] beta Value to assign to the diagonal elements of A.
     * @param[in] A m-by-n matrix.
     * @param[in] lda Leading dimension of A.
     */
    template <class uplo_t, typename TA>
    void laset(
        uplo_t uplo, idx_t m, idx_t n, TA alpha, TA beta, TA* A, idx_t lda)
    {
        using internal::create_matrix;

        // check arguments
        tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
                            uplo != Uplo::General);

        // quick return
        if (m <= 0 || n <= 0) return;

        // Matrix views
        auto A_ = create_matrix<TA>(A, m, n, lda);

        return laset(uplo, alpha, beta, A_);
    }

    /** Initializes a matrix to diagonal and off-diagonal values
     *
     * @param[in] matrixtype :
     *
     *        'U': A is assumed to be upper triangular; elements below the
     * diagonal are not referenced. 'L': A is assumed to be lower triangular;
     * elements above the diagonal are not referenced. otherwise, A is assumed
     * to be a full matrix.
     * @param[in] m Number of rows of A.
     * @param[in] n Number of columns of A.
     * @param[in] alpha Value to be assigned to the off-diagonal elements of A.
     * @param[in] beta Value to assign to the diagonal elements of A.
     * @param[in] A m-by-n matrix.
     * @param[in] lda Leading dimension of A.
     *
     * @see laset( Uplo, idx_t, idx_t, TA, TA, TA*, idx_t )
     *
     * @ingroup legacy_lapack
     */
    template <typename TA>
    void inline laset(MatrixType matrixtype,
                      idx_t m,
                      idx_t n,
                      TA alpha,
                      TA beta,
                      TA* A,
                      idx_t lda)
    {
        if (matrixtype == MatrixType::Upper) {
            laset(UPPER_TRIANGLE, m, n, alpha, beta, A, lda);
        }
        else if (matrixtype == MatrixType::Lower) {
            laset(LOWER_TRIANGLE, m, n, alpha, beta, A, lda);
        }
        else {
            laset(GENERAL, m, n, alpha, beta, A, lda);
        }
    }

}  // namespace legacy
}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LASET_HH
