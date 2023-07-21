/// @file multishift_qr.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqr0.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULTISHIFT_QR_HH
#define TLAPACK_MULTISHIFT_QR_HH

#include <functional>

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/agressive_early_deflation.hpp"
#include "tlapack/lapack/multishift_qr_sweep.hpp"

namespace tlapack {

/**
 * Options struct for multishift_qr
 */
template <TLAPACK_INDEX idx_t = size_t>
struct FrancisOpts {
    // Function that returns the number of shifts to use
    // for a given matrix size
    std::function<idx_t(idx_t, idx_t)> nshift_recommender =
        [](idx_t n, idx_t nh) -> idx_t {
        if (n < 30) return 2;
        if (n < 60) return 4;
        if (n < 150) return 10;
        if (n < 590) return idx_t(n / log2(n));
        if (n < 3000) return 64;
        if (n < 6000) return 128;
        return 256;
    };

    // Function that returns the number of shifts to use
    // for a given matrix size
    std::function<idx_t(idx_t, idx_t)> deflation_window_recommender =
        [](idx_t n, idx_t nh) -> idx_t {
        if (n < 30) return 2;
        if (n < 60) return 4;
        if (n < 150) return 10;
        if (n < 590) return idx_t(n / log2(n));
        if (n < 3000) return 96;
        if (n < 6000) return 192;
        return 384;
    };

    // On exit of the routine. Stores the number of times AED and sweep were
    // called And the total number of shifts used.
    int n_aed = 0;
    int n_sweep = 0;
    int n_shifts_total = 0;
    // Threshold to switch between blocked and unblocked code
    idx_t nmin = 75;
    // Threshold of percent of AED window that must converge to skip a sweep
    idx_t nibble = 14;
};

template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
WorkInfo multishift_qr_worksize_sweep(
    bool want_t,
    bool want_z,
    size_type<matrix_t> ilo,
    size_type<matrix_t> ihi,
    const matrix_t& A,
    const vector_t& w,
    const matrix_t& Z,
    const FrancisOpts<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = ncols(A);
    const idx_t nh = ihi - ilo;
    const idx_t nsr = opts.nshift_recommender(n, nh);
    const auto shifts = slice(w, range{0, nsr});

    return multishift_QR_sweep_worksize<T>(want_t, want_z, ilo, ihi, A, shifts,
                                           Z);
}

/** Worspace query of multishift_qr()
 *
 * @param[in] want_t bool.
 *      If true, the full Schur factor T will be computed.
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 * @param[in] ihi    integer.
 *      The matrix A is assumed to be already quasi-triangular in rows and
 *      columns ihi:n.
 * @param[in] A  n by n matrix.
 * @param w Not referenced.
 * @param[in] Z  n by n matrix.
 *
 * @param[in,out] opts Options.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T,
          TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
WorkInfo multishift_qr_worksize(
    bool want_t,
    bool want_z,
    size_type<matrix_t> ilo,
    size_type<matrix_t> ihi,
    const matrix_t& A,
    const vector_t& w,
    const matrix_t& Z,
    const FrancisOpts<size_type<matrix_t> >& opts = {})
{
    using idx_t = size_type<matrix_t>;

    const idx_t n = ncols(A);

    // quick return
    WorkInfo workinfo;
    if (ilo + 1 >= ihi || n < opts.nmin) return workinfo;

    {
        const idx_t nw_max = (n - 3) / 3;

        idx_t ls = 0, ld = 0;
        workinfo = agressive_early_deflation_worksize<T>(
            want_t, want_z, ilo, ihi, nw_max, A, w, Z, ls, ld, opts);
    }

    workinfo.minMax(multishift_qr_worksize_sweep<T>(want_t, want_z, ilo, ihi, A,
                                                    w, Z, opts));

    return workinfo;
}

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          TLAPACK_RWORKSPACE work_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
int multishift_qr(bool want_t,
                  bool want_z,
                  size_type<matrix_t> ilo,
                  size_type<matrix_t> ihi,
                  matrix_t& A,
                  vector_t& w,
                  matrix_t& Z,
                  work_t& work,
                  FrancisOpts<size_type<matrix_t> >& opts)
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

    int n_aed = 0;
    int n_sweep = 0;
    int n_shifts_total = 0;

    // check arguments
    tlapack_check_false(n != nrows(A));
    tlapack_check_false((idx_t)size(w) != n);
    if (want_z) {
        tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)));
    }

    // quick return
    if (nh <= 0) return 0;
    if (nh == 1) w[ilo] = A(ilo, ilo);

    // Workspace may not be in good shape for multishift_QR_sweep,
    // so reshape, slice and reshape again
    auto work2 = [&]() {
        // Workspace query for multishift_QR_sweep
        WorkInfo workinfo = multishift_qr_worksize_sweep<TA>(
            want_t, want_z, ilo, ihi, A, w, Z, opts);
        const idx_t workSize = nrows(work) * ncols(work);
        auto aux = reshape(work, workSize, 1);
        auto aux2 = slice(aux, range{workSize - workinfo.size(), workSize},
                          range{0, 1});
        return reshape(aux2, workinfo.m, workinfo.n);
    }();

    // Tiny matrices must use lahqr
    if (n < nmin) {
        return lahqr(want_t, want_z, ilo, ihi, A, w, Z);
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

    // nw is the deflation window size
    idx_t nw;

    for (idx_t iter = 0; iter <= itmax; ++iter) {
        if (iter == itmax) {
            // The QR algorithm failed to converge, return with error.
            info = istop;
            break;
        }

        if (ilo + 1 >= istop) {
            if (ilo + 1 == istop) w[ilo] = A(ilo, ilo);
            // All eigenvalues have been found, exit and return 0.
            break;
        }

        // istart is the start of the active subblock. Either
        // istart = ilo, or H(istart, istart-1) = 0. This means
        // that we can treat this subblock separately.
        idx_t istart = ilo;

        // Find active block
        for (idx_t i = istop - 1; i > ilo; --i) {
            if (A(i, i - 1) == zero) {
                istart = i;
                break;
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
        agressive_early_deflation(want_t, want_z, istart, istop, nw, A, w, Z,
                                  ls, ld, work, opts);

        istop = istop - ld;

        if (ld > 0) k_defl = 0;

        // Skip an expensive QR sweep if there is a (partly heuristic)
        // reason to expect that many eigenvalues will deflate without it.
        // Here, the QR sweep is skipped if many eigenvalues have just been
        // deflated or if the remaining active block is small.
        if (ld > 0 and (100 * ld > nwr * nibble or
                        (istop - istart) <= min(nmin, nw_max))) {
            continue;
        }

        k_defl = k_defl + 1;
        idx_t ns = min(nh - 1, min(ls, nsr));

        idx_t i_shifts = istop - ls;

        if (k_defl % non_convergence_limit_shift == 0) {
            ns = nsr;
            for (idx_t i = i_shifts; i < istop - 1; i = i + 2) {
                real_t ss = abs1(A(i, i - 1)) + abs1(A(i - 1, i - 2));
                TA aa = dat1 * ss + A(i, i);
                TA bb = ss;
                TA cc = dat2 * ss;
                TA dd = aa;
                lahqr_eig22(aa, bb, cc, dd, w[i], w[i + 1]);
            }
        }
        else {
            if (ls < nsr / 2) {
                // Got nsr/2 or fewer shifts? Then use multi/double shift qr to
                // get more
                auto temp = slice(A, range{n - nsr, n}, range{0, nsr});
                auto shifts = slice(w, range{istop - nsr, istop});
                auto Z_slice = slice(Z, range{0, nsr}, range{0, nsr});
                int ierr = lahqr(false, false, 0, nsr, temp, shifts, Z_slice);

                ns = nsr - ierr;

                if (ns < 2) {
                    // In case of a rare QR failure, use eigenvalues
                    // of the trailing 2x2 submatrix
                    TA aa = A(istop - 2, istop - 2);
                    TA bb = A(istop - 2, istop - 1);
                    TA cc = A(istop - 1, istop - 2);
                    TA dd = A(istop - 1, istop - 1);
                    lahqr_eig22(aa, bb, cc, dd, w[istop - 2], w[istop - 1]);
                    ns = 2;
                }

                i_shifts = istop - ns;
            }

            // Sort the shifts (helps a little)
            // Bubble sort keeps complex conjugate pairs together
            bool sorted = false;
            idx_t k = istop;
            while (!sorted && k > i_shifts) {
                sorted = true;
                for (idx_t i = i_shifts; i < k - 1; ++i) {
                    if (abs1(w[i]) < abs1(w[i + 1])) {
                        sorted = false;
                        const type_t<vector_t> tmp = w[i];
                        w[i] = w[i + 1];
                        w[i + 1] = tmp;
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
                if (imag(w[i]) != -imag(w[i - 1])) {
                    const type_t<vector_t> tmp = w[i];
                    w[i] = w[i - 1];
                    w[i - 1] = w[i - 2];
                    w[i - 2] = tmp;
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
                if (imag(w[i_shifts]) == zero) {
                    if (abs(real(w[i_shifts]) - A(istop - 1, istop - 1)) <
                        abs(real(w[i_shifts + 1]) - A(istop - 1, istop - 1)))
                        w[i_shifts + 1] = w[i_shifts];
                    else
                        w[i_shifts] = w[i_shifts + 1];
                }
            }
        }
        auto shifts = slice(w, range{i_shifts, i_shifts + ns});

        n_sweep = n_sweep + 1;
        n_shifts_total = n_shifts_total + ns;
        multishift_QR_sweep(want_t, want_z, istart, istop, A, shifts, Z, work2);
    }

    opts.n_aed = n_aed;
    opts.n_shifts_total = n_shifts_total;
    opts.n_sweep = n_sweep;

    return info;
}

template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX work_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
inline int multishift_qr(bool want_t,
                         bool want_z,
                         size_type<matrix_t> ilo,
                         size_type<matrix_t> ihi,
                         matrix_t& A,
                         vector_t& w,
                         matrix_t& Z,
                         work_t& work)
{
    FrancisOpts<size_type<matrix_t> > opts = {};
    return multishift_qr(want_t, want_z, ilo, ihi, A, w, Z, work, opts);
}

/** multishift_qr computes the eigenvalues and optionally the Schur
 *  factorization of an upper Hessenberg matrix, using the multishift
 *  implicit QR algorithm with AED.
 *
 *  The Schur factorization is returned in standard form. For complex matrices
 *  this means that the matrix T is upper-triangular. The diagonal entries
 *  of T are also its eigenvalues. For real matrices, this means that the
 *  matrix T is block-triangular, with real eigenvalues appearing as 1x1 blocks
 *  on the diagonal and imaginary eigenvalues appearing as 2x2 blocks on the
 * diagonal. All 2x2 blocks are normalized so that the diagonal entries are
 * equal to the real part of the eigenvalue.
 *
 *
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * @return  i if the QR algorithm failed to compute all the eigenvalues
 *            elements i:ihi of w contain those eigenvalues which have been
 *            successfully computed.
 *
 * @param[in] want_t bool.
 *      If true, the full Schur factor T will be computed.
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 * @param[in] ihi    integer.
 *      The matrix A is assumed to be already quasi-triangular in rows and
 *      columns ihi:n.
 * @param[in,out] A  n by n matrix.
 *      On entry, the matrix A.
 *      On exit, if info=0 and want_t=true, the Schur factor T.
 *      T is quasi-triangular in rows and columns ilo:ihi, with
 *      the diagonal (block) entries in standard form (see above).
 * @param[out] w  size n vector.
 *      On exit, if info=0, w(ilo:ihi) contains the eigenvalues
 *      of A(ilo:ihi,ilo:ihi). The eigenvalues appear in the same
 *      order as the diagonal (block) entries of T.
 * @param[in,out] Z  n by n matrix.
 *      On entry, the previously calculated Schur factors
 *      On exit, the orthogonal updates applied to A are accumulated
 *      into Z.
 *
 * @param[in,out] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *      - Output parameters
 *          @c opts.n_aed,
 *          @c opts.n_sweep and
 *          @c opts.n_shifts_total
 *      are updated inside the routine.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_SVECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
int multishift_qr(bool want_t,
                  bool want_z,
                  size_type<matrix_t> ilo,
                  size_type<matrix_t> ihi,
                  matrix_t& A,
                  vector_t& w,
                  matrix_t& Z,
                  FrancisOpts<size_type<matrix_t> >& opts)
{
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // Functor
    Create<matrix_t> new_matrix;

    const idx_t n = ncols(A);
    const idx_t nh = ihi - ilo;

    // This routine uses the space below the subdiagonal as workspace
    // For small matrices, this is not enough
    // if n < nmin, the matrix will be passed to lahqr
    const idx_t nmin = opts.nmin;

    // check arguments
    tlapack_check_false(n != nrows(A));
    tlapack_check_false((idx_t)size(w) != n);
    if (want_z) {
        tlapack_check_false((n != ncols(Z)) or (n != nrows(Z)));
    }

    // quick return
    if (nh <= 0) return 0;
    if (nh == 1) w[ilo] = A(ilo, ilo);

    // Tiny matrices must use lahqr
    if (n < nmin) {
        return lahqr(want_t, want_z, ilo, ihi, A, w, Z);
    }

    // Allocates workspace
    WorkInfo workinfo =
        multishift_qr_worksize<TA>(want_t, want_z, ilo, ihi, A, w, Z, opts);
    std::vector<TA> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return multishift_qr(want_t, want_z, ilo, ihi, A, w, Z, work, opts);
}

template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<is_complex<type_t<vector_t> >, int> = 0>
inline int multishift_qr(bool want_t,
                         bool want_z,
                         size_type<matrix_t> ilo,
                         size_type<matrix_t> ihi,
                         matrix_t& A,
                         vector_t& w,
                         matrix_t& Z)
{
    FrancisOpts<size_type<matrix_t> > opts = {};
    return multishift_qr(want_t, want_z, ilo, ihi, A, w, Z, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_MULTISHIFT_QR_HH
