/// @file aggressive_early_deflation_generalized.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_AED_GENERALIZED_HH
#define TLAPACK_AED_GENERALIZED_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/FrancisOpts.hpp"
#include "tlapack/lapack/generalized_schur_move.hpp"
#include "tlapack/lapack/gghrd.hpp"
#include "tlapack/lapack/lahqz.hpp"
#include "tlapack/lapack/lahqz_eig22.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/laset.hpp"

namespace tlapack {

/** @copybrief aggressive_early_deflation()
 * Workspace is provided as an argument.
 * @copydetails aggressive_early_deflation()
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR alpha_t,
          TLAPACK_SVECTOR beta_t>
void aggressive_early_deflation_generalized(bool want_t,
                                            bool want_q,
                                            bool want_z,
                                            size_type<matrix_t> ilo,
                                            size_type<matrix_t> ihi,
                                            size_type<matrix_t> nw,
                                            matrix_t& A,
                                            matrix_t& B,
                                            alpha_t& alpha,
                                            beta_t& beta,
                                            matrix_t& Q,
                                            matrix_t& Z,
                                            size_type<matrix_t>& ns,
                                            size_type<matrix_t>& nd)
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // Constants
    const real_t one(1);
    const real_t zero(0);
    const idx_t n = ncols(A);
    // Because we will use the lower triangular part of A as workspace,
    // We have a maximum window size
    const idx_t nw_max = (n - 3) / 3;
    const real_t eps = ulp<real_t>();
    const real_t small_num = safe_min<real_t>() * ((real_t)n / eps);
    // Size of the deflation window
    const idx_t jw = min(min(nw, ihi - ilo), nw_max);
    // First row index in the deflation window
    const idx_t kwtop = ihi - jw;

    // check arguments
    tlapack_check(nrows(A) == n);
    tlapack_check(ncols(B) == n);
    tlapack_check(nrows(B) == n);
    if (want_q) {
        tlapack_check(ncols(Q) == n);
        tlapack_check(nrows(Q) == n);
    }
    if (want_z) {
        tlapack_check(ncols(Z) == n);
        tlapack_check(nrows(Z) == n);
    }
    tlapack_check((idx_t)size(alpha) == n);

    // s is the value just outside the window. It determines the spike
    // together with the orthogonal schur factors.
    T s_spike;
    if (kwtop == ilo)
        s_spike = zero;
    else
        s_spike = A(kwtop, kwtop - 1);

    if (kwtop + 1 == ihi) {
        // 1x1 deflation window, not much to do
        alpha[kwtop] = A(kwtop, kwtop);
        beta[kwtop] = B(kwtop, kwtop);
        ns = 1;
        nd = 0;
        if (abs1(s_spike) <= max(small_num, eps * abs1(A(kwtop, kwtop)))) {
            ns = 0;
            nd = 1;
            if (kwtop > ilo) A(kwtop, kwtop - 1) = zero;
        }
        return;
        // Note: The max() above may not propagate a NaN in A(kwtop, kwtop).
    }

    // Define workspace matrices
    // We use the lower triangular part of A as workspace
    // AW and WH overlap, but WH is only used after we no longer need
    // AW so it is ok.
    auto Qc = slice(A, range{n - jw, n}, range{0, jw});
    auto Zc = slice(B, range{n - jw, n}, range{0, jw});
    auto Aw = slice(A, range{n - jw, n}, range{jw, 2 * jw});
    auto Bw = slice(B, range{n - jw, n}, range{jw, 2 * jw});
    auto WH = slice(A, range{n - jw, n}, range{jw, n - jw - 3});
    auto WV = slice(A, range{jw + 3, n - jw}, range{0, jw});

    // Convert the window to spike-triangular form. i.e. calculate the
    // Schur form of the deflation window.
    // If the QZ algorithm fails to converge, it can still be
    // partially in Schur form. In that case we continue on a smaller
    // window (note the use of infqz later in the code).
    auto A_window = slice(A, range{kwtop, ihi}, range{kwtop, ihi});
    auto B_window = slice(B, range{kwtop, ihi}, range{kwtop, ihi});
    auto alpha_window = slice(alpha, range{kwtop, ihi});
    auto beta_window = slice(beta, range{kwtop, ihi});
    laset(LOWER_TRIANGLE, zero, zero, Aw);
    laset(LOWER_TRIANGLE, zero, zero, Bw);
    for (idx_t j = 0; j < jw; ++j)
        for (idx_t i = 0; i < min(j + 2, jw); ++i)
            Aw(i, j) = A_window(i, j);
    for (idx_t j = 0; j < jw; ++j)
        for (idx_t i = 0; i < min(j + 1, jw); ++i)
            Bw(i, j) = B_window(i, j);
    laset(GENERAL, zero, one, Qc);
    laset(GENERAL, zero, one, Zc);
    int infqz;
    infqz = lahqz(true, true, true, 0, jw, Aw, Bw, alpha_window, beta_window,
                  Qc, Zc);

    // TODO: use multishift_qz recursively
    // if (jw < (idx_t)opts.nmin)
    //     infqr = lahqr(true, true, 0, jw, TW, s_window, V);
    // else {
    //     infqr =
    //         multishift_qr_work(true, true, 0, jw, TW, s_window, V, work,
    //         opts);
    //     for (idx_t j = 0; j < jw; ++j)
    //         for (idx_t i = j + 2; i < jw; ++i)
    //             TW(i, j) = zero;
    // }

    // Deflation detection loop
    // one eigenvalue block at a time, we will check if it is deflatable
    // by checking the bottom spike element. If it is not deflatable,
    // we move the block up. This moves other blocks down to check.
    ns = jw;
    idx_t ilst = infqz;
    while (ilst < ns) {
        bool bulge = false;
        if (is_real<T>)
            if (ns > 1)
                if (Aw(ns - 1, ns - 2) != zero) bulge = true;

        if (!bulge) {
            // 1x1 eigenvalue block
            real_t foo = abs1(Aw(ns - 1, ns - 1));
            if (foo == zero) foo = abs1(s_spike);
            if (abs1(s_spike) * abs1(Qc(0, ns - 1)) <=
                max(small_num, eps * foo)) {
                // Eigenvalue is deflatable
                ns = ns - 1;
            }
            else {
                // Eigenvalue is not deflatable.
                // Move it up out of the way.
                idx_t ifst = ns - 1;
                generalized_schur_move(true, true, Aw, Bw, Qc, Zc, ifst, ilst);
                ilst = ilst + 1;
            }
            // Note: The max() above may not propagate a NaN in TW(ns-1,ns-1).
        }
        else {
            // 2x2 eigenvalue block
            real_t foo =
                abs(Aw(ns - 1, ns - 1)) +
                sqrt(abs(Aw(ns - 1, ns - 2))) * sqrt(abs(Aw(ns - 2, ns - 1)));
            if (foo == zero) foo = abs(s_spike);
            if (max(abs(s_spike * Qc(0, ns - 1)),
                    abs(s_spike * Qc(0, ns - 2))) <=
                max<real_t>(small_num, eps * foo)) {
                // Eigenvalue pair is deflatable
                ns = ns - 2;
            }
            else {
                // Eigenvalue pair is not deflatable.
                // Move it up out of the way.
                idx_t ifst = ns - 2;
                generalized_schur_move(true, true, Aw, Bw, Qc, Zc, ifst, ilst);
                ilst = ilst + 2;
            }
        }
    }

    if (ns == 0) s_spike = zero;

    if (ns == jw) {
        // Agressive early deflation didn't deflate any eigenvalues
        // We don't need to apply the update to the rest of the matrix
        nd = jw - ns;
        ns = ns - infqz;
        return;
    }

    // Recalculate the eigenvalues
    idx_t i = 0;
    while (i < jw) {
        idx_t n1 = 1;
        if (is_real<T>)
            if (i + 1 < jw)
                if (Aw(i + 1, i) != zero) n1 = 2;

        if (n1 == 1) {
            alpha[kwtop + i] = Aw(i, i);
            beta[kwtop + i] = Bw(i, i);
        }
        else {
            auto A22 = slice(Aw, range(i, i + 2), range(i, i + 2));
            auto B22 = slice(Bw, range(i, i + 2), range(i, i + 2));
            lahqz_eig22(A22, B22, alpha[kwtop + i], alpha[kwtop + i + 1],
                        beta[kwtop + i], beta[kwtop + i + 1]);
        }
        i = i + n1;
    }

    // Reduce A back to Hessenberg form (if neccesary)
    if (s_spike != zero) {
        // Use rotations to remove the spike
        for (idx_t i = ns - 1; i > 0; i--) {
            T t1 = conj(Qc(0, i - 1));
            T t2 = conj(Qc(0, i));
            real_t c;
            T s;
            rotg(t1, t2, c, s);

            auto q1 = col(Qc, i - 1);
            auto q2 = col(Qc, i);
            rot(q1, q2, c, conj(s));
            Qc(0, i) = 0;

            auto a1 = slice(Aw, i - 1, range(0, jw));
            auto a2 = slice(Aw, i, range(0, jw));
            rot(a1, a2, c, s);

            auto b1 = slice(Bw, i - 1, range(0, jw));
            auto b2 = slice(Bw, i, range(0, jw));
            rot(b1, b2, c, s);
        }
        // Remove fill-in from B
        for (idx_t i = ns - 1; i > 0; i--) {
            real_t c;
            T s;
            rotg(Bw(i, i), Bw(i, i - 1), c, s);
            s = -s;
            Bw(i, i - 1) = (T)0.;

            auto b1 = slice(Bw, range(0, i), i - 1);
            auto b2 = slice(Bw, range(0, i), i);
            rot(b1, b2, c, conj(s));

            auto a1 = slice(Aw, range(0, ns), i - 1);
            auto a2 = slice(Aw, range(0, ns), i);
            rot(a1, a2, c, conj(s));

            auto z1 = col(Zc, i - 1);
            auto z2 = col(Zc, i);
            rot(z1, z2, c, conj(s));
        }

        // // Hessenberg-triangular reduction
        gghrd(true, true, 0, ns, Aw, Bw, Qc, Zc);
    }

    // Copy the deflation window back into place
    if (kwtop > 0) A(kwtop, kwtop - 1) = s_spike * conj(Qc(0, 0));
    for (idx_t j = 0; j < jw; ++j)
        for (idx_t i = 0; i < min(j + 2, jw); ++i)
            A(kwtop + i, kwtop + j) = Aw(i, j);
    for (idx_t j = 0; j < jw; ++j)
        for (idx_t i = 0; i < min(j + 1, jw); ++i)
            B(kwtop + i, kwtop + j) = Bw(i, j);

    // Store number of deflated eigenvalues
    nd = jw - ns;
    ns = ns - infqz;

    //
    // Update rest of the matrix using matrix matrix multiplication
    //
    idx_t istart_m, istop_m;
    if (want_t) {
        istart_m = 0;
        istop_m = n;
    }
    else {
        istart_m = ilo;
        istop_m = ihi;
    }
    // Update A
    if (ihi < istop_m) {
        idx_t i = ihi;
        while (i < istop_m) {
            idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
            auto A_slice = slice(A, range{kwtop, ihi}, range{i, i + iblock});
            auto WH_slice =
                slice(WH, range{0, nrows(A_slice)}, range{0, ncols(A_slice)});
            gemm(CONJ_TRANS, NO_TRANS, one, Qc, A_slice, WH_slice);
            lacpy(GENERAL, WH_slice, A_slice);
            i = i + iblock;
        }
    }
    if (istart_m < kwtop) {
        idx_t i = istart_m;
        while (i < kwtop) {
            idx_t iblock = std::min<idx_t>(kwtop - i, nrows(WV));
            auto A_slice = slice(A, range{i, i + iblock}, range{kwtop, ihi});
            auto WV_slice =
                slice(WV, range{0, nrows(A_slice)}, range{0, ncols(A_slice)});
            gemm(NO_TRANS, NO_TRANS, one, A_slice, Zc, WV_slice);
            lacpy(GENERAL, WV_slice, A_slice);
            i = i + iblock;
        }
    }
    // Update B
    if (ihi < istop_m) {
        idx_t i = ihi;
        while (i < istop_m) {
            idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
            auto B_slice = slice(B, range{kwtop, ihi}, range{i, i + iblock});
            auto WH_slice =
                slice(WH, range{0, nrows(B_slice)}, range{0, ncols(B_slice)});
            gemm(CONJ_TRANS, NO_TRANS, one, Qc, B_slice, WH_slice);
            lacpy(GENERAL, WH_slice, B_slice);
            i = i + iblock;
        }
    }
    if (istart_m < kwtop) {
        idx_t i = istart_m;
        while (i < kwtop) {
            idx_t iblock = std::min<idx_t>(kwtop - i, nrows(WV));
            auto B_slice = slice(B, range{i, i + iblock}, range{kwtop, ihi});
            auto WV_slice =
                slice(WV, range{0, nrows(B_slice)}, range{0, ncols(B_slice)});
            gemm(NO_TRANS, NO_TRANS, one, B_slice, Zc, WV_slice);
            lacpy(GENERAL, WV_slice, B_slice);
            i = i + iblock;
        }
    }
    // Update Q
    if (want_q) {
        idx_t i = 0;
        while (i < n) {
            idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
            auto Q_slice = slice(Q, range{i, i + iblock}, range{kwtop, ihi});
            auto WV_slice =
                slice(WV, range{0, nrows(Q_slice)}, range{0, ncols(Q_slice)});
            gemm(NO_TRANS, NO_TRANS, one, Q_slice, Qc, WV_slice);
            lacpy(GENERAL, WV_slice, Q_slice);
            i = i + iblock;
        }
    }
    // Update Z
    if (want_z) {
        idx_t i = 0;
        while (i < n) {
            idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
            auto Z_slice = slice(Z, range{i, i + iblock}, range{kwtop, ihi});
            auto WV_slice =
                slice(WV, range{0, nrows(Z_slice)}, range{0, ncols(Z_slice)});
            gemm(NO_TRANS, NO_TRANS, one, Z_slice, Zc, WV_slice);
            lacpy(GENERAL, WV_slice, Z_slice);
            i = i + iblock;
        }
    }
}  // namespace tlapack

}  // namespace tlapack

#endif  // TLAPACK_AED_GENERALIZED_HH
