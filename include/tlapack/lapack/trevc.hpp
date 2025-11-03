/// @file trevc.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dtrevc.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC_HH
#define TLAPACK_TREVC_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/lapack/trevc3_backsolve.hpp"
#include "tlapack/lapack/trevc3_forwardsolve.hpp"

namespace tlapack {

enum class HowMny : char {
    All = 'A',    ///< all eigenvectors
    Back = 'B',   ///< all eigenvectors, backtransformed by input matrix
    Select = 'S'  ///< selected eigenvectors
};

/**
 * Options struct for trevc3.
 */
struct Trevc3Opts {
    int nb = 64;  ///< Block size
};

/**
 *
 * TREVC computes some or all of the right and/or left eigenvectors of
 * an upper quasi-triangular matrix T.
 * Matrices of this type are produced by the Schur factorization of
 * a general matrix:  A = Q*T*Q**T
 *
 * The right eigenvector x and the left eigenvector y of T corresponding
 * to an eigenvalue w are defined by:
 *
 *    T*x = w*x,     (y**T)*T = w*(y**T)
 *
 * where y**T denotes the transpose of the vector y.
 * The eigenvalues are not input to this routine, but are read directly
 * from the diagonal blocks of T.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 * input matrix. If Q is the orthogonal factor that reduces a matrix
 * A to Schur form T, then Q*X and Q*Y are the matrices of right and
 * left eigenvectors of A.
 *
 * @param[in] side tlapack::Side
 *                 Specifies whether right or left eigenvectors are required:
 *                 = Side::Right: right eigenvectors only;
 *                 = Side::Left: left eigenvectors only;
 *                 = Side::Both: both right and left eigenvectors.
 *
 * @ingroup trevc
 */
template <TLAPACK_SIDE side_t,
          TLAPACK_VECTOR select_t,
          TLAPACK_MATRIX matrix_T_t,
          TLAPACK_MATRIX matrix_Vl_t,
          TLAPACK_MATRIX matrix_Vr_t,
          TLAPACK_WORKSPACE work_t>
int trevc(const side_t side,
          const HowMny howmny,
          select_t& select,
          const matrix_T_t& T,
          matrix_Vl_t& Vl,
          matrix_Vr_t& Vr,
          work_t& work,
          const Trevc3Opts& opts = {})
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<TT>;

    const idx_t n = nrows(T);
    // Number of columns of Vl and Vr
    const idx_t mm = max(ncols(Vl), ncols(Vr));

    const idx_t nb = opts.nb;

    // Quick return
    if (n == 0) return 0;

    idx_t m = 0;  // Actual number of eigenvectors to compute
    if (howmny == HowMny::Select) {
        // Set m to the number of columns required to store the selected
        // eigenvectors.
        // If necessary, the array select is standardized for complex
        // conjugate pairs so that select[j] is true and select[j+1] is false.
        idx_t j = 0;
        while (j < n) {
            bool pair = false;
            if (j < n - 1) {
                if (T(j + 1, j) != TT(0)) {
                    pair = true;
                }
            }
            if (!pair) {
                if (select[j]) {
                    m++;
                }
                j++;
            }
            else {
                if (select[j] || select[j + 1]) {
                    select[j] = true;
                    select[j + 1] = false;
                    m += 2;
                }
                j += 2;
            }
        }
    }
    else {
        m = n;
    }

    // Make sure that the matrices Vl and Vr have enough space
    tlapack_check(mm >= m);

    auto [v1, work2] = reshape(work, n);
    auto [v2, work3] = reshape(work2, n);
    auto [v3, work4] = reshape(work3, n);

    if (side == Side::Right || side == Side::Both) {
        //
        // Compute right eigenvectors.
        //
        idx_t iVr = m - 1;  // current column of Vr to store the eigenvector
        for (idx_t ii = 0; ii < n; ii++) {
            idx_t i = n - 1 - ii;
            if (HowMny::Select == howmny) {
                if (select[i] == false) {
                    continue;
                }
            }
            if (i > 0) {
                if (T(i, i - 1) != TT(0)) {
                    continue;
                }
            }

            bool pair = false;
            if (i < n - 1) {
                if (T(i + 1, i) != TT(0)) {
                    pair = true;
                }
            }

            if (!pair) {
                //
                // Real eigenvalue
                //

                // Calculate eigenvector of the upper quasi-triangular matrix T
                trevc3_backsolve_single(T, v1, i);

                // Backtransform eigenvector if required
                if (howmny == HowMny::Back) {
                    auto Q_slice = slice(Vr, range(0, n), range(0, i + 1));
                    auto v1_slice = slice(v1, range(0, i + 1));
                    gemv(Op::NoTrans, TT(1), Q_slice, v1_slice, v2);
                    for (idx_t k = 0; k < n; ++k) {
                        Vr(k, i) = v2[k];
                    }
                }
                else {
                    // Copy the eigenvector to Vr
                    for (idx_t k = 0; k < n; ++k) {
                        Vr(k, iVr) = v1[k];
                    }
                }
                iVr--;
            }
            else {
                if constexpr (is_real<TT>) {
                    // Complex conjugate pair
                    // Calculate eigenvector of the upper quasi-triangular
                    // matrix T
                    trevc3_backsolve_double(T, v1, v2, i);

                    // Backtransform eigenvector pair if required
                    if (howmny == HowMny::Back) {
                        auto Q_slice1 = slice(Vr, range(0, n), range(0, i + 2));
                        auto v2_slice = slice(v2, range(0, i + 2));
                        gemv(Op::NoTrans, TT(1), Q_slice1, v2_slice, v3);
                        // copy v3 to Vr(:, i+1)
                        for (idx_t k = 0; k < n; ++k) {
                            Vr(k, i + 1) = v3[k];
                        }
                        // Note: we assume that these eigenvectors are
                        // constructed so that v1[i+1] = 0, otherwise, we would
                        // need an extra workspace vector here.
                        auto Q_slice2 = slice(Vr, range(0, n), range(0, i + 1));
                        auto v1_slice = slice(v1, range(0, i + 1));
                        gemv(Op::NoTrans, TT(1), Q_slice2, v1_slice, v3);
                        // copy v3 to Vr(:, i)
                        for (idx_t k = 0; k < n; ++k) {
                            Vr(k, i) = v3[k];
                        }
                    }
                    else {
                        // Copy the eigenvector pair to Vr
                        for (idx_t k = 0; k < n; ++k) {
                            Vr(k, iVr - 1) = v1[k];
                            Vr(k, iVr) = v2[k];
                        }
                    }
                    iVr -= 2;
                }
            }
        }
    }

    if (side == Side::Left || side == Side::Both) {
        //
        // Compute left eigenvectors.
        //
        idx_t iVl = 0;  // current column of Vl to store the eigenvector
        for (idx_t i = 0; i < n; i++) {
            if (HowMny::Select == howmny) {
                if (select[i] == false) {
                    continue;
                }
            }
            if (i > 0) {
                if (T(i, i - 1) != TT(0)) {
                    continue;
                }
            }

            bool pair = false;
            if (i < n - 1) {
                if (T(i + 1, i) != TT(0)) {
                    pair = true;
                }
            }

            if (!pair) {
                //
                // Real eigenvalue
                //

                // Calculate eigenvector of the upper quasi-triangular matrix T
                trevc3_forwardsolve_single(T, v1, i);

                // Backtransform eigenvector if required
                if (howmny == HowMny::Back) {
                    auto Q_slice = slice(Vl, range(0, n), range(i, n));
                    auto v1_slice = slice(v1, range(i, n));
                    gemv(Op::NoTrans, TT(1), Q_slice, v1_slice, v2);
                    for (idx_t k = 0; k < n; ++k) {
                        Vl(k, i) = v2[k];
                    }
                }
                else {
                    // Copy the eigenvector to Vl
                    for (idx_t k = 0; k < n; ++k) {
                        Vl(k, iVl) = v1[k];
                    }
                }
                iVl++;
            }
            else {
                if constexpr (is_real<TT>) {
                    // Complex conjugate pair
                    // Calculate eigenvector of the upper quasi-triangular
                    // matrix T
                    trevc3_forwardsolve_double(T, v1, v2, i);

                    // Backtransform eigenvector pair if required
                    if (howmny == HowMny::Back) {
                        auto Q_slice1 = slice(Vl, range(0, n), range(i, n));
                        auto v1_slice = slice(v1, range(i, n));
                        gemv(Op::NoTrans, TT(1), Q_slice1, v1_slice, v3);
                        // copy v3 to Vl(:, i)
                        for (idx_t k = 0; k < n; ++k) {
                            Vl(k, i) = v3[k];
                        }
                        // Note: we assume that these eigenvectors are
                        // constructed so that v2[i] = 0, otherwise, we would
                        // need an extra workspace vector here.
                        auto Q_slice2 = slice(Vl, range(0, n), range(i + 1, n));
                        auto v2_slice = slice(v2, range(i + 1, n));
                        gemv(Op::NoTrans, TT(1), Q_slice2, v2_slice, v3);
                        // copy v3 to Vl(:, i+1)
                        for (idx_t k = 0; k < n; ++k) {
                            Vl(k, i + 1) = v3[k];
                        }
                    }
                    else {
                        // Copy the eigenvector pair to Vl
                        for (idx_t k = 0; k < n; ++k) {
                            Vl(k, iVl) = v1[k];
                            Vl(k, iVl + 1) = v2[k];
                        }
                    }
                    iVl += 2;
                }
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC_HH
