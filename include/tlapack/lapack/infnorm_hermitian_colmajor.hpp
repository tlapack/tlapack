/// @file infnorm_hermitian_colmajor.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_INFNORM_HE_COLMAJOR_HH
#define TLAPACK_INFNORM_HE_COLMAJOR_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Worspace query of infnorm_hermitian_colmajor()
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in] A n-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
constexpr WorkInfo infnorm_hermitian_colmajor_worksize(uplo_t uplo,
                                                       const matrix_t& A)
{
    return WorkInfo(nrows(A));
}

/** @copybrief infnorm_hermitian_colmajor()
 * Workspace is provided as an argument.
 * @copydetails infnorm_hermitian_colmajor()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_MATRIX matrix_t,
          TLAPACK_WORKSPACE work_t>
auto infnorm_hermitian_colmajor_work(uplo_t uplo,
                                     const matrix_t& A,
                                     work_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);

    // constants
    const idx_t n = nrows(A);

    // quick return
    if (n <= 0) return real_t(0);

    // Reshape workspace and extract w
    WorkInfo workinfo = infnorm_hermitian_colmajor_worksize<T>(A);
    auto W = reshape(work, workinfo.m, workinfo.n);
    auto w = slice(W, range{0, n}, 0);

    // Norm value
    real_t norm(0);

    for (idx_t i = 0; i < n; ++i)
        w[i] = T(0);

    if (uplo == Uplo::Upper) {
        for (idx_t j = 0; j < n; ++j) {
            real_t sum(0);
            for (idx_t i = 0; i < j; ++i) {
                const real_t absa = abs(A(i, j));
                sum += absa;
                w[i] += absa;
            }
            w[j] = sum + abs(real(A(j, j)));
        }
        for (idx_t i = 0; i < n; ++i) {
            real_t sum = w[i];
            if (sum > norm)
                norm = sum;
            else {
                if (isnan(sum)) return sum;
            }
        }
    }
    else {
        for (idx_t j = 0; j < n; ++j) {
            real_t sum = w[j] + abs(real(A(j, j)));
            for (idx_t i = j + 1; i < n; ++i) {
                const real_t absa = abs(A(i, j));
                sum += absa;
                w[i] += absa;
            }
            if (sum > norm)
                norm = sum;
            else {
                if (isnan(sum)) return sum;
            }
        }
    }

    return norm;
}

/** Calculates the infinity norm of a column-major hermitian matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 *
 * @see lanhe() for the generic implementation that does not use workspaces.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in] A n-by-n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
auto infnorm_hermitian_colmajor(uplo_t uplo, const matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;

    // check arguments
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);

    // constants
    const idx_t n = nrows(A);

    // quick return
    if (n <= 0) return real_t(0);

    // Allocates workspace
    WorkInfo workinfo = infnorm_hermitian_colmajor_worksize<T>(uplo, A);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return infnorm_hermitian_colmajor_work(uplo, A, work);
}

}  // namespace tlapack

#endif  // TLAPACK_INFNORM_HE_COLMAJOR_HH
