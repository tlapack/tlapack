/// @file gebd2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgebd2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEBD2_HH
#define TLAPACK_GEBD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gebd2().
 *
 * @param[in] A m-by-n matrix.
 *      On entry, the m by n general matrix to be reduced.
 *
 * @param tauv Not referenced.
 *
 * @param tauw Not referenced.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo gebd2_worksize(const matrix_t& A,
                                  const vector_t& tauv,
                                  const vector_t& tauw)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    WorkInfo workinfo;
    if (n > 1) {
        auto&& A11 = cols(A, range{1, n});
        workinfo = larf_worksize<T>(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE,
                                    col(A, 0), tauv[0], A11);

        if (m > 1) {
            auto&& B11 = rows(A11, range{1, m});
            auto&& row0 = slice(A, 0, range{1, n});

            workinfo.minMax(larf_worksize<T>(
                RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, row0, tauw[0], B11));
        }
    }

    return workinfo;
}

/** @copybrief gebd2()
 * Workspace is provided as an argument.
 * @copydetails gebd2()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int gebd2_work(matrix_t& A, vector_t& tauv, vector_t& tauw, work_t& work)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check_false((idx_t)size(tauv) < min(m, n));
    tlapack_check_false((idx_t)size(tauw) < min(m, n));

    // quick return
    if (n <= 0) return 0;

    if (m >= n) {
        //
        // Reduce to upper bidiagonal form
        //
        for (idx_t j = 0; j < n; ++j) {
            // Generate elementary reflector H(j) to annihilate A(j+1:m,j)
            auto v = slice(A, range(j, m), j);
            larfg(FORWARD, COLUMNWISE_STORAGE, v, tauv[j]);

            if (j < n - 1) {
                // Apply H(j)**H to A(j:m,j+1:n) from the left
                auto A11 = slice(A, range(j, m), range(j + 1, n));
                larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v,
                          conj(tauv[j]), A11, work);

                // Generate elementary reflector G(j) to annihilate A(j,j+2:n)
                auto w = slice(A, j, range(j + 1, n));
                larfg(FORWARD, ROWWISE_STORAGE, w, tauw[j]);

                // Apply G(j) to A(j+1:m,j+1:n) from the right
                if (j < m - 1) {
                    auto B11 = slice(A, range(j + 1, m), range(j + 1, n));
                    larf_work(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, w, tauw[j],
                              B11, work);
                }
            }
            else {
                tauw[j] = T(0);
            }
        }
    }
    else {
        //
        // Reduce to lower bidiagonal form
        //
        for (idx_t j = 0; j < m; ++j) {
            // Generate elementary reflector G(j) to annihilate A(j,j+1:n)
            auto w = slice(A, j, range(j, n));
            larfg(FORWARD, ROWWISE_STORAGE, w, tauw[j]);

            if (j < m - 1) {
                // Apply G(j) to A(j+1:m,j:n) from the right
                auto A11 = slice(A, range(j + 1, m), range(j, n));
                larf_work(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, w, tauw[j], A11,
                          work);

                // Generate elementary reflector H(j) to annihilate A(j+2:m,j)
                auto v = slice(A, range(j + 1, m), j);
                larfg(FORWARD, COLUMNWISE_STORAGE, v, tauv[j]);

                // Apply H(j)**H to A(j+1:m,j+1:n) from the left
                if (j < n - 1) {
                    auto B11 = slice(A, range(j + 1, m), range(j + 1, n));
                    larf_work(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, v,
                              conj(tauv[j]), B11, work);
                }
            }
            else {
                tauv[j] = T(0);
            }
        }
    }

    return 0;
}

/** Reduces a general m by n matrix A to an upper
 *  real bidiagonal form B by a unitary transformation:
 * \[
 *          Q**H * A * Z = B.
 * \]
 *
 * The matrices Q and Z are represented as products of elementary
 * reflectors:
 *
 * If m >= n,
 * \[
 *          Q = H(1) H(2) . . . H(n)  and  Z = G(1) G(2) . . . G(n-1)
 * \]
 * Each H(i) and G(i) has the form:
 * \[
 *          H(j) = I - tauv * v * v**H  and G(j) = I - tauw * w * w**H
 * \]
 * where tauv and tauw are complex scalars, and v and w are complex
 * vectors; v(1:j-1) = 0, v(j) = 1, and v(j+1:m) is stored on exit in
 * A(j+1:m,j); w(1:j) = 0, w(j+1) = 1, and w(j+2:n) is stored on exit in
 * A(j,i+2:n); tauv is stored in tauv(j) and tauw in tauw(j).
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the m by n general matrix to be reduced.
 *      - if m >= n, the diagonal and the first superdiagonal
 *        are overwritten with the upper bidiagonal matrix B; the
 *        elements below the diagonal, with the array tauv, represent
 *        the unitary matrix Q as a product of elementary reflectors,
 *        and the elements above the first superdiagonal, with the array
 *        tauw, represent the unitary matrix Z as a product of elementary
 *        reflectors.
 *      - if m < n, the diagonal and the first superdiagonal
 *        are overwritten with the lower bidiagonal matrix B; the
 *        elements below the first subdiagonal, with the array tauv, represent
 *        the unitary matrix Q as a product of elementary reflectors,
 *        and the elements above the diagonal, with the array tauw, represent
 *        the unitary matrix Z as a product of elementary reflectors.
 *
 * @param[out] tauv Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Q.
 *
 * @param[out] tauw Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors which
 *      represent the unitary matrix Z.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gebd2(matrix_t& A, vector_t& tauv, vector_t& tauw)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // Functors
    Create<matrix_t> new_matrix;

    // constants
    const idx_t n = ncols(A);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = gebd2_worksize<T>(A, tauv, tauw);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gebd2_work(A, tauv, tauw, work);
}

}  // namespace tlapack

#endif  // TLAPACK_GEBD2_HH
