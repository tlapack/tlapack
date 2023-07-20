/// @file lange.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lange.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LANGE_HH
#define TLAPACK_LANGE_HH

#include "tlapack/lapack/lassq.hpp"

namespace tlapack {

/** Worspace query of lange()
 *
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each
 * column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each
 * row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 *
 * @param[in] A m-by-n matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_NORM norm_t, TLAPACK_SMATRIX matrix_t>
inline constexpr WorkInfo lange_worksize(norm_t normType, const matrix_t& A)
{
    return WorkInfo(0);
}

/** Worspace query of lange()
 *
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each
 * column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each
 * row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_NORM norm_t, TLAPACK_MATRIX matrix_t>
inline constexpr WorkInfo lange_worksize(norm_t normType, const matrix_t& A)
{
    if constexpr (is_same_v<T, type_t<matrix_t>>)
        if (normType == Norm::Inf) return WorkInfo(nrows(A));

    return WorkInfo(0);
}

/** Calculates the norm of a matrix.
 *
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 *
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each
 * column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each
 * row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 *
 * @param[in] A m-by-n matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_NORM norm_t, TLAPACK_SMATRIX matrix_t>
auto lange(norm_t normType, const matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(normType != Norm::Fro && normType != Norm::Inf &&
                        normType != Norm::Max && normType != Norm::One);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Norm value
    real_t norm(0);

    if (normType == Norm::Max) {
        for (idx_t j = 0; j < n; ++j) {
            for (idx_t i = 0; i < m; ++i) {
                real_t temp = abs(A(i, j));

                if (temp > norm)
                    norm = temp;
                else {
                    if (isnan(temp)) return temp;
                }
            }
        }
    }
    else if (normType == Norm::Inf) {
        for (idx_t i = 0; i < m; ++i) {
            real_t sum(0);
            for (idx_t j = 0; j < n; ++j)
                sum += abs(A(i, j));

            if (sum > norm)
                norm = sum;
            else {
                if (isnan(sum)) return sum;
            }
        }
    }
    else if (normType == Norm::One) {
        for (idx_t j = 0; j < n; ++j) {
            real_t sum(0);
            for (idx_t i = 0; i < m; ++i)
                sum += abs(A(i, j));

            if (sum > norm)
                norm = sum;
            else {
                if (isnan(sum)) return sum;
            }
        }
    }
    else {
        real_t scale(0), sum(1);
        for (idx_t j = 0; j < n; ++j)
            lassq(col(A, j), scale, sum);
        norm = scale * sqrt(sum);
    }

    return norm;
}

/** Calculates the norm of a matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 *
 * @param[in] normType
 *      - Norm::Max: Maximum absolute value over all elements of the matrix.
 *          Note: this is not a consistent matrix norm.
 *      - Norm::One: 1-norm, the maximum value of the absolute sum of each
 * column.
 *      - Norm::Inf: Inf-norm, the maximum value of the absolute sum of each
 * row.
 *      - Norm::Fro: Frobenius norm of the matrix.
 *          Square root of the sum of the square of each entry in the matrix.
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_NORM norm_t, TLAPACK_MATRIX matrix_t>
auto lange(norm_t normType, const matrix_t& A)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false(normType != Norm::Fro && normType != Norm::Inf &&
                        normType != Norm::Max && normType != Norm::One);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // redirect for max-norm, one-norm and Frobenius norm
    if (normType == Norm::Max)
        return lange(MAX_NORM, A);
    else if (normType == Norm::One)
        return lange(ONE_NORM, A);
    else if (normType == Norm::Fro)
        return lange(FROB_NORM, A);
    else if (normType == Norm::Inf) {
        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
        // the infinite norm.

        // Allocates workspace
        WorkInfo workinfo = lange_worksize<T>(normType, A);
        std::vector<T> work_;
        auto work = new_matrix(work_, workinfo.m, workinfo.n);
        auto w = slice(work, range{0, m}, 0);

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
}

}  // namespace tlapack

#endif  // TLAPACK_LANGE_HH
