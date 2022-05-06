/// @file multishift_qr.hpp
/// @author Thijs, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaqr0.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __MULTISHIFT_QR_HH__
#define __MULTISHIFT_QR_HH__

#include <complex>
#include <vector>
#include <functional>

#include "legacy_api/base/utils.hpp"
#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/multishift_qr_sweep.hpp"
#include "lapack/agressive_early_deflation.hpp"

namespace tlapack
{

    /**
     * Options struct for multishift_qr
     */
    template <typename idx_t, typename T>
    struct francis_opts_t
    {

        // Function that returns the number of shifts to use
        // for a given matrix size
        std::function<idx_t(idx_t, idx_t)> nshift_recommender = [](idx_t n, idx_t nh) -> idx_t
        {
            if (n < 30)
                return 2;
            if (n < 60)
                return 4;
            if (n < 150)
                return 10;
            if (n < 590)
                return 16;
            if (n < 3000)
                return 64;
            if (n < 6000)
                return 128;
            return 256;
        };

        // Function that returns the number of shifts to use
        // for a given matrix size
        std::function<idx_t(idx_t, idx_t)> deflation_window_recommender = [](idx_t n, idx_t nh) -> idx_t
        {
            if (n < 30)
                return 2;
            if (n < 60)
                return 4;
            if (n < 150)
                return 10;
            if (n < 590)
                return 16;
            if (n < 3000)
                return 96;
            if (n < 6000)
                return 192;
            return 384;
        };

        idx_t nibble = 14;
        // Workspace pointer, if no workspace is provided, one will be allocated internally
        T *_work = nullptr;
        // Workspace size
        idx_t lwork;
    };

    /** multishift_qr computes the eigenvalues and optionally the Schur
     *  factorization of an upper Hessenberg matrix, using the multishift
     *  implicit QR algorithm with AED.
     *
     *  The Schur factorization is returned in standard form. For complex matrices
     *  this means that the matrix T is upper-triangular. The diagonal entries
     *  of T are also its eigenvalues. For real matrices, this means that the
     *  matrix T is block-triangular, with real eigenvalues appearing as 1x1 blocks
     *  on the diagonal and imaginary eigenvalues appearing as 2x2 blocks on the diagonal.
     *  All 2x2 blocks are normalized so that the diagonal entries are equal to the real part
     *  of the eigenvalue.
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
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    int multishift_qr(bool want_t, bool want_z, size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &w, matrix_t &Z, const francis_opts_t<size_type<matrix_t>, type_t<matrix_t>> &opts = {})
    {
        using TA = type_t<matrix_t>;
        using real_t = real_type<TA>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;

        // constants
        const real_t rzero(0);
        const TA one(1);
        const TA zero(0);
        const real_t eps = uroundoff<real_t>();
        const real_t small_num = safe_min<real_t>() / uroundoff<real_t>();
        const idx_t non_convergence_limit_window = 5;
        const idx_t non_convergence_limit_shift = 6;
        const real_t dat1 = 3.0 / 4.0;
        const real_t dat2 = -0.4375;

        const idx_t n = ncols(A);
        const idx_t nh = ihi - ilo;

        // This routine uses the space below the subdiagonal as workspace
        // For small matrices, this is not enough
        // if n < nmin, the matrix will be passed to lahqr
        const idx_t nmin = 15;

        // Recommended number of shifts
        const idx_t nsr = opts.nshift_recommender(n, nh);

        // Recommended deflation window size
        const idx_t nwr = opts.deflation_window_recommender(n, nh);
        const idx_t nw_max = (n - 3) / 3;

        const idx_t nibble = opts.nibble;

        // check arguments
        lapack_error_if(n != nrows(A), -5);
        lapack_error_if((idx_t)size(w) != n, -6);
        if (want_z)
        {
            lapack_error_if((n != ncols(Z)) or (n != nrows(Z)), -7);
        }

        // quick return
        if (nh <= 0)
            return 0;
        if (nh == 1)
            w[ilo] = A(ilo, ilo);

        // Tiny matrices must use lahqr
        if (n < nmin)
        {
            return lahqr(want_t, want_z, ilo, ihi, A, w, Z);
        }

        // Get the workspace
        TA *_work;
        idx_t lwork;
        // idx_t required_workspace = get_work_multishift_qr(want_t, want_z, ilo, ihi, A, w, Z, opts);
        idx_t required_workspace = 3 * (nsr);
        // Store whether or not a workspace was locally allocated
        bool locally_allocated = false;
        if (opts._work and required_workspace <= opts.lwork)
        {
            // Provided workspace is large enough, use it
            _work = opts._work;
            lwork = opts.lwork;
        }
        else
        {
            // No workspace provided or not large enough, allocate it
            locally_allocated = true;
            lwork = required_workspace;
            _work = new TA[lwork];
        }
        auto V = legacyMatrix<TA, layout<matrix_t>>(3, nsr, &_work[0], layout<matrix_t> == Layout ::ColMajor ? 3 : nsr);

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

        for (idx_t iter = 0; iter <= itmax; ++iter)
        {

            if (iter == itmax)
            {
                // The QR algorithm failed to converge, return with error.
                info = istop;
                break;
            }

            if (ilo + 1 >= istop)
            {
                // All eigenvalues have been found, exit and return 0.
                break;
            }

            // istart is the start of the active subblock. Either
            // istart = ilo, or H(istart, istart-1) = 0. This means
            // that we can treat this subblock separately.
            idx_t istart = ilo;

            // Find active block
            for (idx_t i = istop - 1; i > ilo; --i)
            {
                if (A(i, i - 1) == zero)
                {
                    istart = i;
                    break;
                }
            }
            //
            // Agressive early deflation
            //
            idx_t nh = istop - istart;
            idx_t nwupbd = std::min(nh, nw_max);
            if (k_defl < non_convergence_limit_window)
            {
                nw = std::min(nwupbd, nwr);
            }
            else
            {
                // There have been no deflations in many iterations
                // Try to vary the deflation window size.
                nw = std::min(nwupbd, 2 * nw);
            }
            if( nh <= 4 ){
                // n >= nmin, so there is always enough space for a 4x4 window
                nw = nh;
            }
            if (nw < nw_max)
            {
                if (nw + 1 >= nh)
                    nw = nh;
                idx_t kwtop = ihi - nw;
                if (abs1(A(kwtop, kwtop - 1)) > abs1(A(kwtop - 1, kwtop - 2)))
                    nw = nw + 1;
            }

            idx_t ls, ld;
            agressive_early_deflation(want_t, want_z, istart, istop, nw, A, w, Z, ls, ld);

            istop = istop - ld;

            if (ld > 0)
                k_defl = 0;

            // Skip an expensive QR sweep if there is a (partly heuristic)
            // reason to expect that many eigenvalues will deflate without it.
            // Here, the QR sweep is skipped if many eigenvalues have just been
            // deflated or if the remaining active block is small.
            if (ld > 0 and (100 * ld > nwr * nibble or (istop - istart) <= nw_max))
            {
                continue;
            }

            k_defl = k_defl + 1;
            idx_t ns = std::min(nh - 1, std::min(ls, nsr));

            if (k_defl % non_convergence_limit_shift == 0)
            {
                ns = nsr;
                for (idx_t i = istop - ns; i < istop - 1; i = i + 2)
                {
                    real_t ss = abs1(A(i, i - 1)) + abs1(A(i - 1, i - 2));
                    TA aa = dat1 * ss + A(i, i);
                    TA bb = ss;
                    TA cc = dat2 * ss;
                    TA dd = aa;
                    lahqr_eig22(aa, bb, cc, dd, w[i], w[i + 1]);
                }
            }

            idx_t i_shifts = istop - ls;

            // Sort the shifts (helps a little)
            // Bubble sort keeps complex conjugate pairs together
            bool sorted = false;
            idx_t k = istop;
            while (!sorted && k > i_shifts)
            {
                sorted = true;
                for (idx_t i = i_shifts; i < k - 1; ++i)
                {
                    if (abs1(w[i]) > abs1(w[i + 1]))
                    {
                        sorted = false;
                        auto tmp = w[i];
                        w[i] = w[i + 1];
                        w[i + 1] = tmp;
                    }
                }
                --k;
            }

            // If there are only two shifts and both are real
            // then use only one (helps avoid interference)
            if (!is_complex<TA>::value)
            {
                if (ns == 2)
                {
                    if (imag(w[i_shifts]) == zero)
                    {
                        if (abs(real(w[i_shifts]) - A(istop - 1, istop - 1)) < abs(real(w[i_shifts + 1]) - A(istop - 1, istop - 1)))
                            w[i_shifts + 1] = w[i_shifts];
                        else
                            w[i_shifts] = w[i_shifts + 1];
                    }
                }
            }

            i_shifts = istop - ns;
            auto shifts = slice(w, pair{i_shifts, istop});

            multishift_QR_sweep(want_t, want_z, istart, istop, A, shifts, Z, V);
        }

        if (locally_allocated)
            delete _work;
        return info;
    }

} // lapack

#endif // __MULTISHIFT_QR_HH__
