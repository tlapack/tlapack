/// @file infnorm_colmajor.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_INFNORM_COLMAJOR_HH
#define TLAPACK_INFNORM_COLMAJOR_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Worspace query of infnorm_colmajor()
 *
 * @param[in] A m-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_MATRIX matrix_t>
constexpr WorkInfo infnorm_colmajor_worksize(const matrix_t& A)
{
    return WorkInfo(nrows(A));
}

/** @copybrief infnorm_colmajor()
 * Workspace is provided as an argument.
 * @copydetails infnorm_colmajor()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_WORKSPACE work_t>
auto infnorm_colmajor_work(const matrix_t& A, work_t& work)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Reshape workspace and extract w
    WorkInfo workinfo = infnorm_colmajor_worksize<T>(A);
    auto W = reshape(work, workinfo.m, workinfo.n);
    auto w = slice(W, range{0, m}, 0);

    // Norm value
    real_t norm(0);

    for (idx_t i = 0; i < m; ++i)
        w[i] = abs(A(i, 0));

    for (idx_t j = 1; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            w[i] += abs(A(i, j));

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

/** Calculates the infinity norm of a column-major matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 *
 * @see lange() for the generic implementation that does not use workspaces.
 *
 * @param[in] A m-by-n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t>
auto infnorm_colmajor(const matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Allocates workspace
    WorkInfo workinfo = infnorm_colmajor_worksize<T>(A);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return infnorm_colmajor_work(A, work);
}

}  // namespace tlapack

#endif  // TLAPACK_INFNORM_COLMAJOR_HH
