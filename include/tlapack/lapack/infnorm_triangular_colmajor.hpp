/// @file infnorm_triangular_colmajor.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_INFNORM_TR_COLMAJOR_HH
#define TLAPACK_INFNORM_TR_COLMAJOR_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Worspace query of infnorm_triangular_colmajor()
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrix_t>
constexpr WorkInfo infnorm_triangular_colmajor_worksize(uplo_t uplo,
                                                        diag_t diag,
                                                        const matrix_t& A)
{
    return WorkInfo(nrows(A));
}

/** Calculates the infinity norm of a column-major triangular matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 *
 * @see lantr() for the generic implementation that does not use workspaces.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n matrix.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrix_t,
          TLAPACK_WORKSPACE work_t>
auto infnorm_triangular_colmajor_work(uplo_t uplo,
                                      diag_t diag,
                                      const matrix_t& A,
                                      work_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Slice workspace
    auto w = slice(work, range{0, m}, 0);

    // Norm value
    real_t norm(0);

    for (idx_t i = 0; i < n; ++i)
        w[i] = T(0);

    if (uplo == Uplo::Upper) {
        if (diag == Diag::NonUnit) {
            for (idx_t i = 0; i < m; ++i)
                w[i] = real_t(0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i <= min(j, m - 1); ++i)
                    w[i] += abs(A(i, j));
        }
        else {
            for (idx_t i = 0; i < m; ++i)
                w[i] = real_t(1);

            for (idx_t j = 1; j < n; ++j) {
                for (idx_t i = 0; i < min(j, m); ++i)
                    w[i] += abs(A(i, j));
            }
        }
    }
    else {
        if (diag == Diag::NonUnit) {
            for (idx_t i = 0; i < m; ++i)
                w[i] = real_t(0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j; i < m; ++i)
                    w[i] += abs(A(i, j));
        }
        else {
            for (idx_t i = 0; i < min(m, n); ++i)
                w[i] = real_t(1);
            for (idx_t i = n; i < m; ++i)
                w[i] = real_t(0);

            for (idx_t j = 1; j < n; ++j) {
                for (idx_t i = j + 1; i < m; ++i)
                    w[i] += abs(A(i, j));
            }
        }
    }

    for (idx_t i = 0; i < m; ++i) {
        real_t temp = w[i];

        if (temp > norm)
            norm = temp;
        else {
            if (isnan(temp)) return temp;
        }
    }

    return norm;
}

/** Calculates the infinity norm of a column-major triangular matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 *
 * @see lantr() for the generic implementation that does not use workspaces.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_DIAG diag_t, TLAPACK_MATRIX matrix_t>
auto infnorm_triangular_colmajor(uplo_t uplo, diag_t diag, const matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Allocates workspace
    WorkInfo workinfo = infnorm_triangular_colmajor_worksize<T>(uplo, A);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return infnorm_triangular_colmajor_work(uplo, A, work);
}

}  // namespace tlapack

#endif  // TLAPACK_INFNORM_TR_COLMAJOR_HH
