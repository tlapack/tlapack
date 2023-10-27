/// @file multishift_qz.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqz0.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULTISHIFT_QZ_HH
#define TLAPACK_MULTISHIFT_QZ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/FrancisOpts.hpp"
#include "tlapack/lapack/aggressive_early_deflation_generalized.hpp"
#include "tlapack/lapack/multishift_qz_sweep.hpp"

namespace tlapack {

/** @copybrief multishift_qr()
 * Workspace is provided as an argument.
 * @copydetails multishift_qr()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR alpha_t,
          TLAPACK_SVECTOR beta_t>
int multishift_qz(bool want_t,
                  bool want_q,
                  bool want_z,
                  size_type<matrix_t> ilo,
                  size_type<matrix_t> ihi,
                  matrix_t& A,
                  matrix_t& B,
                  alpha_t& alpha,
                  beta_t& beta,
                  matrix_t& Q,
                  matrix_t& Z,
                  FrancisOpts& opts)
{
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const real_t zero(0);
    const idx_t non_convergence_limit_window = 5;
    const idx_t non_convergence_limit_shift = 6;
    const real_t dat1(0.75);
    const real_t dat2(-0.4375);

    const idx_t n = ncols(A);
    const idx_t nh = ihi - ilo;

    // This routine uses the space below the subdiagonal as workspace
    // For small matrices, this is not enough
    // if n < nmin, the matrix will be passed to lahqr
    const idx_t nmin = opts.nmin;

    // Recommended number of shifts
    const idx_t nsr = opts.nshift_recommender(n, nh);

    // Recommended deflation window size
    const idx_t nwr = opts.deflation_window_recommender(n, nh);
    const idx_t nw_max = (n - 3) / 3;

    const idx_t nibble = opts.nibble;

    const real_t eps = ulp<real_t>();
    const real_t small_num = safe_min<real_t>();

    int n_aed = 0;
    int n_sweep = 0;
    int n_shifts_total = 0;

    // check arguments
    tlapack_check_false(n != nrows(A));
    tlapack_check_false(n != ncols(B));
    tlapack_check_false(n != nrows(B));
    tlapack_check_false((idx_t)size(alpha) != n);
    tlapack_check_false((idx_t)size(beta) != n);
    if (want_z) {
        tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)));
    }
    if (want_q) {
        tlapack_check_false((n != ncols(Q)) or (n != nrows(Q)));
    }

    // quick return
    if (nh <= 0) return 0;
    if (nh == 1) {
        alpha[ilo] = A(ilo, ilo);
        beta[ilo] = B(ilo, ilo);
    }

    // Tiny matrices must use lahqz
    if (n < nmin) {
        return lahqz(want_t, want_q, want_z, ilo, ihi, A, B, alpha, beta, Q, Z);
    }

    // itmax is the total number of QR iterations allowed.
    // For most matrices, 3 shifts per eigenvalue is enough, so
    // we set itmax to 30 times nh as a safe limit.
    const idx_t itmax = 30 * std::max<idx_t>(10, nh);

    // k_defl counts the number of iterations since a deflation
    idx_t k_defl = 0;

    // istop is the end of the active subblock.
    // As more and more eigenvalues converge, it eventually
    // becomes ilo+1 and the loop ends.
    idx_t istop = ihi;

    int info = 0;

    // Norm of B, used for checking infinite eigenvalues
    const real_t bnorm =
        lange(FROB_NORM, slice(B, range(ilo, ihi), range(ilo, ihi)));

    // nw is the deflation window size
    idx_t nw;

    for (idx_t iter = 0; iter <= itmax; ++iter) {
        if (iter == itmax) {
            // The QZ algorithm failed to converge, return with error.
            info = istop;
            break;
        }

        if (ilo + 1 >= istop) {
            if (ilo + 1 == istop) {
                alpha[ilo] = A(ilo, ilo);
                beta[ilo] = B(ilo, ilo);
            }
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // istart is the start of the active subblock. Either
        // istart = ilo, or H(istart, istart-1) = 0. This means
        // that we can treat this subblock separately.
        idx_t istart = ilo;

        // Find active block
        idx_t istart_m;
        idx_t istop_m;
        if (!want_t) {
            istart_m = istart;
            istop_m = istop;
        }
        else {
            istart_m = 0;
            istop_m = n;
        }

        // Check if active subblock has split
        for (idx_t i = istop - 1; i > istart; --i) {
            if (abs1(A(i, i - 1)) <= small_num) {
                // A(i,i-1) is negligible, take i as new istart.
                A(i, i - 1) = zero;
                istart = i;
                break;
            }

            real_t tst = abs1(A(i - 1, i - 1)) + abs1(A(i, i));
            if (tst == zero) {
                if (i >= ilo + 2) {
                    tst = tst + abs(A(i - 1, i - 2));
                }
                if (i < ihi) {
                    tst = tst + abs(A(i + 1, i));
                }
            }
            if (abs1(A(i, i - 1)) <= eps * tst) {
                //
                // The elementwise deflation test has passed
                // The following performs second deflation test due
                // to Steel, Vandebril and Langou (2023). It has better
                // mathematical foundation and improves accuracy in some
                // examples.
                //
                // The test is |A(i,i-1)|*|A(i-1,i)*B(i,i) - A(i,i)*B(i-1,i)| <=
                // eps*|A(i,i)|*|A(i-1,i-1)*B(i,i)-A(i,i)*B(i-1,i-1)|.
                // The multiplications might overflow so we do some scaling
                // first.
                //
                const real_t tst1 =
                    abs1(B(i, i) * A(i - 1, i) - A(i, i) * B(i - 1, i));
                const real_t tst2 =
                    abs1(B(i, i) * A(i - 1, i - 1) - A(i, i) * B(i - 1, i - 1));
                real_t ab = max(abs1(A(i, i - 1)), tst1);
                real_t ba = min(abs1(A(i, i - 1)), tst1);
                real_t aa = max(abs1(A(i, i)), tst2);
                real_t bb = min(abs1(A(i, i)), tst2);
                real_t s = aa + ab;
                if (ba * (ab / s) <= max(small_num, eps * (bb * (aa / s)))) {
                    // A(i,i-1) is negligible, take i as new istart.
                    A(i, i - 1) = zero;
                    istart = i;
                    break;
                }
            }
        }

        // check for infinite eigenvalues
        for (idx_t i = istart; i < istop; ++i) {
            if (abs1(B(i, i)) <= max(small_num, eps * bnorm)) {
                // B(i,i) is negligible, so B is singular, i.e. (A,B) has an
                // infinite eigenvalue. Move it to the top to be deflated
                B(i, i) = zero;

                real_t c;
                TA s;
                for (idx_t j = i; j > istart; j--) {
                    rotg(B(j - 1, j), B(j - 1, j - 1), c, s);
                    B(j - 1, j - 1) = zero;
                    // Apply rotation from the right
                    {
                        auto b1 = slice(B, range(istart_m, j - 1), j);
                        auto b2 = slice(B, range(istart_m, j - 1), j - 1);
                        rot(b1, b2, c, s);

                        auto a1 = slice(A, range(istart_m, min(j + 2, n)), j);
                        auto a2 =
                            slice(A, range(istart_m, min(j + 2, n)), j - 1);
                        rot(a1, a2, c, s);

                        if (want_z) {
                            auto z1 = col(Z, j);
                            auto z2 = col(Z, j - 1);
                            rot(z1, z2, c, s);
                        }
                    }
                    // Remove fill-in in A
                    if (j < istop - 1) {
                        rotg(A(j, j - 1), A(j + 1, j - 1), c, s);
                        A(j + 1, j - 1) = zero;

                        auto a1 = slice(A, j, range(j, istop_m));
                        auto a2 = slice(A, j + 1, range(j, istop_m));
                        rot(a1, a2, c, s);
                        auto b1 = slice(B, j, range(j + 1, istop_m));
                        auto b2 = slice(B, j + 1, range(j + 1, istop_m));
                        rot(b1, b2, c, s);

                        if (want_q) {
                            auto q1 = col(Q, j);
                            auto q2 = col(Q, j + 1);
                            rot(q1, q2, c, conj(s));
                        }
                    }
                }

                if (istart + 1 < istop) {
                    rotg(A(istart, istart), A(istart + 1, istart), c, s);
                    A(istart + 1, istart) = zero;

                    auto a1 = slice(A, istart, range(istart + 1, istop_m));
                    auto a2 = slice(A, istart + 1, range(istart + 1, istop_m));
                    rot(a1, a2, c, s);
                    auto b1 = slice(B, istart, range(istart + 1, istop_m));
                    auto b2 = slice(B, istart + 1, range(istart + 1, istop_m));
                    rot(b1, b2, c, s);

                    if (want_q) {
                        auto q1 = col(Q, istart);
                        auto q2 = col(Q, istart + 1);
                        rot(q1, q2, c, conj(s));
                    }
                }
                alpha[istart] = A(istart, istart);
                beta[istart] = zero;
                istart = istart + 1;
            }
        }

        if (istart == istop) {
            istop = istop - 1;
            istart = ilo;
            continue;
        }
        // Check if 1x1 block has split off
        if (istart + 1 == istop) {
            k_defl = 0;
            // Normalize the block, make sure B(istart, istart) is real and
            // positive
            if constexpr (is_real<TA>) {
                if (B(istart, istart) < 0.) {
                    for (idx_t i = istart_m; i <= istart; ++i) {
                        B(i, istart) = -B(i, istart);
                        A(i, istart) = -A(i, istart);
                    }
                    if (want_z) {
                        for (idx_t i = 0; i < n; ++i) {
                            Z(i, istart) = -Z(i, istart);
                        }
                    }
                }
            }
            else {
                real_t absB = abs(B(istart, istart));
                if (absB > small_num and (imag(B(istart, istart)) != zero or
                                          real(B(istart, istart)) < zero)) {
                    TA scal = conj(B(istart, istart) / absB);
                    for (idx_t i = istart_m; i <= istart; ++i) {
                        B(i, istart) = scal * B(i, istart);
                        A(i, istart) = scal * A(i, istart);
                    }
                    if (want_z) {
                        for (idx_t i = 0; i < n; ++i) {
                            Z(i, istart) = scal * Z(i, istart);
                        }
                    }
                    B(istart, istart) = absB;
                }
                else {
                    B(istart, istart) = zero;
                }
            }
            alpha[istart] = A(istart, istart);
            beta[istart] = B(istart, istart);
            istop = istart;
            istart = ilo;
            continue;
        }
        // Check if 2x2 block has split off
        if constexpr (is_real<TA>) {
            if (istart + 2 == istop) {
                // 2x2 block, normalize the block
                auto A22 = slice(A, range(istart, istop), range(istart, istop));
                auto B22 = slice(B, range(istart, istop), range(istart, istop));
                lahqz_eig22(A22, B22, alpha[istart], alpha[istart + 1],
                            beta[istart], beta[istart + 1]);
                // Only split off the block if the eigenvalues are imaginary
                if (imag(alpha[istart]) != zero) {
                    // Standardize, that is, rotate so that
                    //     ( B11  0  )
                    // B = (         ) with B11 non-negative
                    //     (  0  B22 )
                    TA ssmin, ssmax, csl, snl, csr, snr;
                    svd22(B22(0, 0), B22(0, 1), B22(1, 1), ssmin, ssmax, csl,
                          snl, csr, snr);

                    if (ssmax < (TA)0) {
                        csr = -csr;
                        snr = -snr;
                        ssmin = -ssmin;
                        ssmax = -ssmax;
                    }

                    B22(0, 0) = ssmax;
                    B22(1, 1) = ssmin;
                    B22(0, 1) = (TA)0;

                    // Apply rotations to A
                    auto a1l = slice(A, istart, range(istart, istop_m));
                    auto a2l = slice(A, istart + 1, range(istart, istop_m));
                    rot(a1l, a2l, csl, snl);
                    auto a1r = slice(A, range(istart_m, istop), istart);
                    auto a2r = slice(A, range(istart_m, istop), istart + 1);
                    rot(a1r, a2r, csr, snr);
                    // Apply rotations to B
                    if (istart + 2 < n) {
                        auto b1l = slice(B, istart, range(istart + 2, istop_m));
                        auto b2l =
                            slice(B, istart + 1, range(istart + 2, istop_m));
                        rot(b1l, b2l, csl, snl);
                    }
                    auto b1r = slice(B, range(istart_m, istart), istart);
                    auto b2r = slice(B, range(istart_m, istart), istart + 1);
                    rot(b1r, b2r, csr, snr);

                    // Apply rotation to Q
                    if (want_q) {
                        auto q1 = col(Q, istart);
                        auto q2 = col(Q, istart + 1);
                        rot(q1, q2, csl, snl);
                    }

                    // Apply rotation to Z
                    if (want_z) {
                        auto z1 = col(Z, istart);
                        auto z2 = col(Z, istart + 1);
                        rot(z1, z2, csr, snr);
                    }

                    k_defl = 0;
                    istop = istart;
                    istart = ilo;
                    continue;
                }
            }
        }

        //
        // Agressive early deflation
        //
        idx_t nh = istop - istart;
        idx_t nwupbd = min(nh, nw_max);
        if (k_defl < non_convergence_limit_window) {
            nw = min(nwupbd, nwr);
        }
        else {
            // There have been no deflations in many iterations
            // Try to vary the deflation window size.
            nw = min(nwupbd, 2 * nw);
        }
        if (nh <= 4) {
            // n >= nmin, so there is always enough space for a 4x4 window
            nw = nh;
        }
        if (nw < nw_max) {
            if (nw + 1 >= nh) nw = nh;
            idx_t kwtop = istop - nw;
            if (kwtop > istart + 2)
                if (abs1(A(kwtop, kwtop - 1)) > abs1(A(kwtop - 1, kwtop - 2)))
                    nw = nw + 1;
        }

        idx_t ls, ld;
        n_aed = n_aed + 1;
        aggressive_early_deflation_generalized(want_t, want_q, want_z, istart,
                                               istop, nw, A, B, alpha, beta, Q,
                                               Z, ls, ld, opts);

        istop = istop - ld;

        if (ld > 0) k_defl = 0;

        // Skip an expensive QR sweep if there is a (partly heuristic)
        // reason to expect that many eigenvalues will deflate without it.
        // Here, the QR sweep is skipped if many eigenvalues have just been
        // deflated or if the remaining active block is small.
        if (ld > 0 and
            (100 * ld > nwr * nibble or istop <= istart + min(nmin, nw_max))) {
            continue;
        }

        k_defl = k_defl + 1;
        idx_t ns = min(nh - 1, min(ls, nsr));

        idx_t i_shifts = istop - ls;

        if (k_defl % non_convergence_limit_shift == 0) {
            // This exceptional shift closely resembles the QR exceptional shift
            // This is nice for easy maintenance, but is not guaranteed to work.
            ns = nsr;
            for (idx_t i = i_shifts; i < istop - 1; i = i + 2) {
                real_t ss = abs1(A(i, i - 1));
                if (i > 1) ss += abs1(A(i - 1, i - 2));
                TA aa = dat1 * ss + A(i, i);
                TA bb = ss;
                TA cc = dat2 * ss;
                TA dd = aa;
                lahqr_eig22(aa, bb, cc, dd, alpha[i], alpha[i + 1]);
                beta[i] = (TA)1;
                beta[i + 1] = (TA)1;
            }
        }
        else {
            // Sort the shifts (helps a little)
            // Bubble sort keeps complex conjugate pairs together
            bool sorted = false;
            idx_t k = istop;
            while (!sorted && k > i_shifts) {
                sorted = true;
                for (idx_t i = i_shifts; i < k - 1; ++i) {
                    if (abs1(alpha[i] * beta[i + 1]) <
                        abs1(alpha[i + 1] * beta[i])) {
                        sorted = false;
                        const type_t<alpha_t> tmp = alpha[i];
                        alpha[i] = alpha[i + 1];
                        alpha[i + 1] = tmp;
                        const type_t<beta_t> tmp2 = beta[i];
                        beta[i] = beta[i + 1];
                        beta[i + 1] = tmp2;
                    }
                }
                --k;
            }

            // Shuffle shifts into pairs of real shifts
            // and pairs of complex conjugate shifts
            // assuming complex conjugate shifts are
            // already adjacent to one another. (Yes,
            // they are.)
            for (idx_t i = istop - 1; i > i_shifts + 1; i = i - 2) {
                if (imag(alpha[i]) != -imag(alpha[i - 1])) {
                    const type_t<alpha_t> tmp = alpha[i];
                    alpha[i] = alpha[i - 1];
                    alpha[i - 1] = alpha[i - 2];
                    alpha[i - 2] = tmp;
                    const type_t<beta_t> tmp2 = beta[i];
                    beta[i] = beta[i - 1];
                    beta[i - 1] = beta[i - 2];
                    beta[i - 2] = tmp2;
                }
            }

            // Since we shuffled the shifts, we will only drop
            // Real shifts
            if (ns % 2 == 1) ns = ns - 1;
            i_shifts = istop - ns;
        }

        // If there are only two shifts and both are real
        // then use only one (helps avoid interference)
        if (is_real<TA>) {
            if (ns == 2) {
                if (imag(alpha[i_shifts]) == zero) {
                    if (abs(real(alpha[i_shifts]) - A(istop - 1, istop - 1)) <
                        abs(real(alpha[i_shifts + 1]) -
                            A(istop - 1, istop - 1))) {
                        alpha[i_shifts + 1] = alpha[i_shifts];
                        beta[i_shifts + 1] = beta[i_shifts];
                    }
                    else {
                        alpha[i_shifts] = alpha[i_shifts + 1];
                        beta[i_shifts] = beta[i_shifts + 1];
                    }
                }
            }
        }
        auto alpha_shifts = slice(alpha, range{i_shifts, i_shifts + ns});
        auto beta_shifts = slice(beta, range{i_shifts, i_shifts + ns});

        n_sweep = n_sweep + 1;
        n_shifts_total = n_shifts_total + ns;
        multishift_QZ_sweep(want_t, want_q, want_z, istart, istop, A, B,
                            alpha_shifts, beta_shifts, Q, Z);
    }

    opts.n_aed = n_aed;
    opts.n_shifts_total = n_shifts_total;
    opts.n_sweep = n_sweep;

    return info;
}

}  // namespace tlapack

#endif  // TLAPACK_MULTISHIFT_QZ_HH
