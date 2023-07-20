/// @file lantr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lantr.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LANTR_HH
#define TLAPACK_LANTR_HH

#include "tlapack/lapack/lassq.hpp"

namespace tlapack {

/** Worspace query of lantr().
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
 * @param[in] uplo
 *      - Uplo::Upper: A is a upper triangle matrix;
 *      - Uplo::Lower: A is a lower triangle matrix.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n triangular matrix.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_NORM norm_t,
          TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_SMATRIX matrix_t>
inline constexpr WorkInfo lantr_worksize(norm_t normType,
                                         uplo_t uplo,
                                         diag_t diag,
                                         const matrix_t& A)
{
    return WorkInfo(0);
}

/** Worspace query of lantr().
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
 * @param[in] uplo
 *      - Uplo::Upper: A is a upper triangle matrix;
 *      - Uplo::Lower: A is a lower triangle matrix.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n triangular matrix.
 *
 * @param[in] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_NORM norm_t,
          TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrix_t>
inline constexpr WorkInfo lantr_worksize(norm_t normType,
                                         uplo_t uplo,
                                         diag_t diag,
                                         const matrix_t& A)
{
    if constexpr (is_same_v<T, type_t<matrix_t>>)
        if (normType == Norm::Inf) return WorkInfo(nrows(A));

    return WorkInfo(0);
}

/** Calculates the norm of a symmetric matrix.
 *
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * @tparam diag_t Either Diag or any class that implements `operator Diag()`.
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
 * @param[in] uplo
 *      - Uplo::Upper: A is a upper triangle matrix;
 *      - Uplo::Lower: A is a lower triangle matrix.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n triangular matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_NORM norm_t,
          TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_SMATRIX matrix_t>
auto lantr(norm_t normType, uplo_t uplo, diag_t diag, const matrix_t& A)
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
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // Norm value
    real_t norm(0);

    if (normType == Norm::Max) {
        if (diag == Diag::NonUnit) {
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i <= min(j, m - 1); ++i) {
                        real_t temp = abs(A(i, j));

                        if (temp > norm)
                            norm = temp;
                        else {
                            if (isnan(temp)) return temp;
                        }
                    }
                }
            }
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j; i < m; ++i) {
                        real_t temp = abs(A(i, j));

                        if (temp > norm)
                            norm = temp;
                        else {
                            if (isnan(temp)) return temp;
                        }
                    }
                }
            }
        }
        else {
            norm = real_t(1);
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < min(j, m); ++i) {
                        real_t temp = abs(A(i, j));

                        if (temp > norm)
                            norm = temp;
                        else {
                            if (isnan(temp)) return temp;
                        }
                    }
                }
            }
            else {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j + 1; i < m; ++i) {
                        real_t temp = abs(A(i, j));

                        if (temp > norm)
                            norm = temp;
                        else {
                            if (isnan(temp)) return temp;
                        }
                    }
                }
            }
        }
    }
    else if (normType == Norm::Inf) {
        if (uplo == Uplo::Upper) {
            for (idx_t i = 0; i < m; ++i) {
                real_t sum(0);
                if (diag == Diag::NonUnit)
                    for (idx_t j = i; j < n; ++j)
                        sum += abs(A(i, j));
                else {
                    sum = real_t(1);
                    for (idx_t j = i + 1; j < n; ++j)
                        sum += abs(A(i, j));
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) return sum;
                }
            }
        }
        else {
            for (idx_t i = 0; i < m; ++i) {
                real_t sum(0);
                if (diag == Diag::NonUnit || i >= n)
                    for (idx_t j = 0; j <= min(i, n - 1); ++j)
                        sum += abs(A(i, j));
                else {
                    sum = real_t(1);
                    for (idx_t j = 0; j < i; ++j)
                        sum += abs(A(i, j));
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) return sum;
                }
            }
        }
    }
    else if (normType == Norm::One) {
        if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j) {
                real_t sum(0);
                if (diag == Diag::NonUnit || j >= m)
                    for (idx_t i = 0; i <= min(j, m - 1); ++i)
                        sum += abs(A(i, j));
                else {
                    sum = real_t(1);
                    for (idx_t i = 0; i < j; ++i)
                        sum += abs(A(i, j));
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) return sum;
                }
            }
        }
        else {
            for (idx_t j = 0; j < n; ++j) {
                real_t sum(0);
                if (diag == Diag::NonUnit)
                    for (idx_t i = j; i < m; ++i)
                        sum += abs(A(i, j));
                else {
                    sum = real_t(1);
                    for (idx_t i = j + 1; i < m; ++i)
                        sum += abs(A(i, j));
                }

                if (sum > norm)
                    norm = sum;
                else {
                    if (isnan(sum)) return sum;
                }
            }
        }
    }
    else {
        real_t scale(1), sum(0);

        if (uplo == Uplo::Upper) {
            if (diag == Diag::NonUnit) {
                for (idx_t j = 0; j < n; ++j)
                    lassq(slice(A, range(0, min(j + 1, m)), j), scale, sum);
            }
            else {
                sum = real_t(min(m, n));
                for (idx_t j = 1; j < n; ++j)
                    lassq(slice(A, range(0, min(j, m)), j), scale, sum);
            }
        }
        else {
            if (diag == Diag::NonUnit) {
                for (idx_t j = 0; j < min(m, n); ++j)
                    lassq(slice(A, range(j, m), j), scale, sum);
            }
            else {
                sum = real_t(min(m, n));
                for (idx_t j = 0; j < min(m - 1, n); ++j)
                    lassq(slice(A, range(j + 1, m), j), scale, sum);
            }
        }
        norm = scale * sqrt(sum);
    }

    return norm;
}

/** Calculates the norm of a triangular matrix.
 *
 * Code optimized for the infinity norm on column-major layouts using a
 * workspace of size at least m, where m is the number of rows of A.
 * @see lantr( norm_t normType, uplo_t uplo, diag_t diag, const matrix_t& A ).
 *
 * @tparam norm_t Either Norm or any class that implements `operator Norm()`.
 * @tparam uplo_t Either Uplo or any class that implements `operator Uplo()`.
 * @tparam diag_t Either Diag or any class that implements `operator Diag()`.
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
 * @param[in] uplo
 *      - Uplo::Upper: A is a upper triangle matrix;
 *      - Uplo::Lower: A is a lower triangle matrix.
 *
 * @param[in] diag
 *     Whether A has a unit or non-unit diagonal:
 *     - Diag::Unit:    A is assumed to be unit triangular.
 *     - Diag::NonUnit: A is not assumed to be unit triangular.
 *
 * @param[in] A m-by-n triangular matrix.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_NORM norm_t,
          TLAPACK_UPLO uplo_t,
          TLAPACK_DIAG diag_t,
          TLAPACK_MATRIX matrix_t>
auto lantr(norm_t normType, uplo_t uplo, diag_t diag, const matrix_t& A)
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
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);

    // quick return
    if (m == 0 || n == 0) return real_t(0);

    // redirect for max-norm, one-norm and Frobenius norm
    if (normType == Norm::Max)
        return lantr(MAX_NORM, uplo, diag, A);
    else if (normType == Norm::One)
        return lantr(ONE_NORM, uplo, diag, A);
    else if (normType == Norm::Fro)
        return lantr(FROB_NORM, uplo, diag, A);
    else if (normType == Norm::Inf) {
        // the code below uses a workspace and is meant for column-major layout
        // so as to do one pass on the data in a contiguous way when computing
        // the infinite norm.

        // Allocates workspace
        WorkInfo workinfo = lantr_worksize<T>(normType, uplo, diag, A);
        std::vector<T> work_;
        auto work = new_matrix(work_, workinfo.m, workinfo.n);
        auto w = slice(work, range{0, n}, 0);

        // Norm value
        real_t norm(0);

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
}

}  // namespace tlapack

#endif  // TLAPACK_LANTR_HH
