/// @file agressive_early_deflation.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_AED_HH
#define TLAPACK_AED_HH

#include "tlapack/base/utils.hpp"

#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/lahqr.hpp"
#include "tlapack/lapack/lahqr_eig22.hpp"
#include "tlapack/lapack/gehd2.hpp"
#include "tlapack/lapack/unghr.hpp"
#include "tlapack/lapack/multishift_qr.hpp"
#include "tlapack/lapack/schur_move.hpp"
#include "tlapack/lapack/schur_swap.hpp"
#include "tlapack/lapack/unmhr.hpp"

namespace tlapack
{
    // Forward declaration
    template < class idx_t > struct francis_opts_t;

    /** Worspace query of agressive_early_deflation().
     *
     * @param[in] want_t bool.
     *      If true, the full Schur factor T will be computed.
     *
     * @param[in] want_z bool.
     *      If true, the Schur vectors Z will be computed.
     *
     * @param[in] ilo    integer.
     *      Either ilo=0 or A(ilo,ilo-1) = 0.
     *
     * @param[in] ihi    integer.
     *      ilo and ihi determine an isolated block in A.
     *
     * @param[in] nw    integer.
     *      Desired window size to perform agressive early deflation on.
     *      If the matrix is not large enough to provide the scratch space
     *      or if the isolated block is small, a smaller value may be used.
     *
     * @param[in] A  n by n matrix.
     *       Hessenberg matrix on which AED will be performed
     *
     * @param s  Not referenced.
     *
     * @param[in] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors.
     *
     * @param ns    Not referenced.
     *
     * @param nd    Not referenced.
     *
     * @param[in] opts Options.
     * 
     * @param[in,out] workinfo
     *      On output, the amount workspace required. It is larger than or equal
     *      to that given on input.
     *
     * @ingroup workspace_query
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<
            is_complex< type_t<vector_t> >::value
        , int > = 0
    >
    void agressive_early_deflation_worksize(
        bool want_t, 
        bool want_z, 
        size_type<matrix_t> ilo, 
        size_type<matrix_t> ihi, 
        size_type<matrix_t> nw, 
        const matrix_t &A, 
        const vector_t &s, 
        const matrix_t &Z, 
        const size_type<matrix_t> &ns,
        const size_type<matrix_t> &nd,
        workinfo_t& workinfo,
        const francis_opts_t< size_type<matrix_t> > &opts = {} )
    {
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;
        
        const idx_t n = ncols(A);
        const idx_t nw_max = (n - 3) / 3;
        const idx_t jw = min( min(nw, ihi - ilo), nw_max );

        auto TW = slice(A, pair{0,jw}, pair{0,jw});

        // quick return
        if( n < 9 || nw <= 1 || ihi <= 1+ilo )
            return;

        if( jw >= opts.nmin )
        {
            auto s_window = slice(s, pair{0,jw});
            auto V = slice(A, pair{0,jw}, pair{0,jw});
            
            multishift_qr_worksize(true, true, 0, jw, TW, s_window, V, workinfo, opts);
        }

        if( jw != ihi-ilo )
        {
            // Hessenberg reduction
            auto tau = slice(A, pair{0,jw}, 0);
            gehrd_worksize(0, jw, TW, tau, workinfo);
        }
    }

    /** agressive_early_deflation accepts as input an upper Hessenberg matrix
     *  H and performs an orthogonal similarity transformation
     *  designed to detect and deflate fully converged eigenvalues from
     *  a trailing principal submatrix.  On output H has been over-
     *  written by a new Hessenberg matrix that is a perturbation of
     *  an orthogonal similarity transformation of H.  It is to be
     *  hoped that the final version of H has many zero subdiagonal
     *  entries.
     *
     * @param[in] want_t bool.
     *      If true, the full Schur factor T will be computed.
     *
     * @param[in] want_z bool.
     *      If true, the Schur vectors Z will be computed.
     *
     * @param[in] ilo    integer.
     *      Either ilo=0 or A(ilo,ilo-1) = 0.
     *
     * @param[in] ihi    integer.
     *      ilo and ihi determine an isolated block in A.
     *
     * @param[in] nw    integer.
     *      Desired window size to perform agressive early deflation on.
     *      If the matrix is not large enough to provide the scratch space
     *      or if the isolated block is small, a smaller value may be used.
     *
     * @param[in,out] A  n by n matrix.
     *       Hessenberg matrix on which AED will be performed
     *
     * @param[out] s  size n vector.
     *      On exit, the entries s[ihi-nd-ns:ihi-nd] contain the unconverged
     *      eigenvalues that can be used a shifts. The entries s[ihi-nd:ihi]
     *      contain the converged eigenvalues. Entries outside the range
     *      s[ihi-nw:ihi] are not changed. The converged shifts are stored
     *      in the same positions as their correspinding diagonal elements
     *      in A.
     *
     * @param[in,out] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors
     *      On exit, the orthogonal updates applied to A accumulated
     *      into Z.
     *
     * @param[out] ns    integer.
     *      Number of eigenvalues available as shifts in s.
     *
     * @param[out] nd    integer.
     *      Number of converged eigenvalues available as shifts in s.
     *
     * @param[in,out] opts Options.
     *      - @c opts.work is used if whenever it has sufficient size.
     *        The sufficient size can be obtained through a workspace query.
     *      - Output parameters
     *          @c opts.n_aed,
     *          @c opts.n_sweep and
     *          @c opts.n_shifts_total
     *        are updated by the internal call to multishift_qr.
     *
     * @ingroup computational
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<
            is_complex< type_t<vector_t> >::value
        , int > = 0
    >
    void agressive_early_deflation(
        bool want_t, 
        bool want_z, 
        size_type<matrix_t> ilo, 
        size_type<matrix_t> ihi, 
        size_type<matrix_t> nw, 
        matrix_t &A, 
        vector_t &s, 
        matrix_t &Z, 
        size_type<matrix_t> &ns, 
        size_type<matrix_t> &nd, 
        francis_opts_t< size_type<matrix_t> > &opts )
    {

        using T = type_t<matrix_t>;
        using real_t = real_type<T>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;
        using std::max;
        using std::min;

        // Constants
        const real_t one(1);
        const real_t zero(0);
        const idx_t n = ncols(A);
        // Because we will use the lower triangular part of A as workspace,
        // We have a maximum window size
        const idx_t nw_max = (n - 3) / 3;
        const real_t eps = ulp<real_t>();
        const real_t small_num = safe_min<real_t>() * ((real_t)n / ulp<real_t>());
        // Size of the deflation window
        const idx_t jw = min(min(nw, ihi - ilo), nw_max);
        // First row index in the deflation window
        const idx_t kwtop = ihi - jw;

        // Assertions
        assert(nrows(A) == n);
        if(want_z) {
            assert(ncols(Z) == n);
            assert(nrows(Z) == n);
        }
        assert((idx_t) size(s) == n);

        // s is the value just outside the window. It determines the spike
        // together with the orthogonal schur factors.
        T s_spike;
        if (kwtop == ilo)
            s_spike = zero;
        else
            s_spike = A(kwtop, kwtop - 1);

        if (kwtop + 1 == ihi)
        {
            // 1x1 deflation window, not much to do
            s[kwtop] = A(kwtop, kwtop);
            ns = 1;
            nd = 0;
            if (abs1(s_spike) <= max(small_num, eps * abs1(A(kwtop, kwtop))))
            {
                ns = 0;
                nd = 1;
                if (kwtop > ilo)
                    A(kwtop, kwtop - 1) = zero;
            }
            return;
        }

        // Allocates workspace
        vectorOfBytes localworkdata;
        Workspace work = [&]()
        {
            workinfo_t workinfo;
            agressive_early_deflation_worksize( want_t, want_z, ilo, ihi, nw, A, s, Z, ns, nd, workinfo, opts );
            return alloc_workspace( localworkdata, workinfo, opts.work );
        }();
        
        // Options to forward
        opts.work = work;
        auto&& gehrdOpts = gehrd_opts_t<idx_t>{ work };

        // Define workspace matrices
        // We use the lower triangular part of A as workspace
        // TW and WH overlap, but WH is only used after we no longer need
        // TW so it is ok.
        auto V = slice(A, pair{n - jw, n}, pair{0, jw});
        auto TW = slice(A, pair{n - jw, n}, pair{jw, 2 * jw});
        auto WH = slice(A, pair{n - jw, n}, pair{jw, n - jw - 3});
        auto WV = slice(A, pair{jw + 3, n - jw}, pair{0, jw});

        // Convert the window to spike-triangular form. i.e. calculate the
        // Schur form of the deflation window.
        // If the QR algorithm fails to convergence, it can still be
        // partially in Schur form. In that case we continue on a smaller
        // window (note the use of infqr later in the code).
        auto A_window = slice(A, pair{kwtop, ihi}, pair{kwtop, ihi});
        auto s_window = slice(s, pair{kwtop, ihi});
        laset(Uplo::Lower, zero, zero, TW);
        for (idx_t j = 0; j < jw; ++j)
            for (idx_t i = 0; i < std::min(j + 2, jw); ++i)
                TW(i, j) = A_window(i, j);
        laset(Uplo::General, zero, one, V);
        int infqr;
        if( jw < opts.nmin )
            infqr = lahqr(true, true, 0, jw, TW, s_window, V);
        else{
            infqr = multishift_qr(true, true, 0, jw, TW, s_window, V, opts);
            for (idx_t j = 0; j < jw; ++j)
                for (idx_t i = j+2; i < jw; ++i)
                    TW(i, j) = zero;
        }

        // Deflation detection loop
        // one eigenvalue block at a time, we will check if it is deflatable
        // by checking the bottom spike element. If it is not deflatable,
        // we move the block up. This moves other blocks down to check.
        ns = jw;
        idx_t ilst = infqr;
        while (ilst < ns)
        {
            bool bulge = false;
            if (!is_complex<T>::value)
                if (ns > 1)
                    if (TW(ns - 1, ns - 2) != zero)
                        bulge = true;

            if (!bulge)
            {
                // 1x1 eigenvalue block
                real_t foo = abs1(TW(ns - 1, ns - 1));
                if (foo == zero)
                    foo = abs1(s_spike);
                if (abs1(s_spike) * abs1(V(0, ns - 1)) <= max(small_num, eps * foo))
                {
                    // Eigenvalue is deflatable
                    ns = ns - 1;
                }
                else
                {
                    // Eigenvalue is not deflatable.
                    // Move it up out of the way.
                    idx_t ifst = ns - 1;
                    schur_move(true, TW, V, ifst, ilst);
                    ilst = ilst + 1;
                }
            }
            else
            {
                // 2x2 eigenvalue block
                real_t foo = tlapack::abs(TW(ns - 1, ns - 1)) + sqrt(tlapack::abs(TW(ns - 1, ns - 2))) * sqrt(tlapack::abs(TW(ns - 2, ns - 1)));
                if (foo == zero)
                    foo = tlapack::abs(s_spike);
                if (max(tlapack::abs(s_spike * V(0, ns - 1)), tlapack::abs(s_spike * V(0, ns - 2))) <= max<real_t>(small_num, eps * foo))
                {
                    // Eigenvalue pair is deflatable
                    ns = ns - 2;
                }
                else
                {
                    // Eigenvalue pair is not deflatable.
                    // Move it up out of the way.
                    idx_t ifst = ns - 2;
                    schur_move(true, TW, V, ifst, ilst);
                    ilst = ilst + 2;
                }
            }
        }

        if (ns == 0)
            s_spike = zero;


        if (ns == jw)
        {
            // Agressive early deflation didn't deflate any eigenvalues
            // We don't need to apply the update to the rest of the matrix
            nd = jw - ns;
            ns = ns - infqr;
            return;
        }

        // sorting diagonal blocks of T improves accuracy for graded matrices.
        // Bubble sort deals well with exchange failures.
        bool sorted = false;
        // Window to be checked (other eigenvalue are sorted)
        idx_t sorting_window_size = jw;
        while (!sorted)
        {
            sorted = true;

            // Index of last eigenvalue that was swapped
            idx_t ilst = 0;

            // Index of the first block
            idx_t i1 = ns;

            while (i1 + 1 < sorting_window_size)
            {

                // Size of the first block
                idx_t n1 = 1;
                if (!is_complex<T>::value)
                    if (TW(i1 + 1, i1) != zero)
                        n1 = 2;

                // Check if there is a next block
                if (i1 + n1 == jw)
                {
                    ilst = ilst - n1;
                    break;
                }

                // Index of the second block
                idx_t i2 = i1 + n1;

                // Size of the second block
                idx_t n2 = 1;
                if (!is_complex<T>::value)
                    if (i2 + 1 < jw)
                        if (TW(i2 + 1, i2) != zero)
                            n2 = 2;

                real_t ev1, ev2;
                if (n1 == 1)
                    ev1 = abs1(TW(i1, i1));
                else
                    ev1 = tlapack::abs(TW(i1, i1)) + sqrt(tlapack::abs(TW(i1 + 1, i1))) * sqrt(tlapack::abs(TW(i1, i1 + 1)));
                if (n2 == 1)
                    ev2 = abs1(TW(i2, i2));
                else
                    ev2 = tlapack::abs(TW(i2, i2)) + sqrt(tlapack::abs(TW(i2 + 1, i2))) * sqrt(tlapack::abs(TW(i2, i2 + 1)));

                if (ev1 > ev2)
                {
                    i1 = i2;
                }
                else
                {
                    sorted = false;
                    int ierr = schur_swap(true, TW, V, i1, n1, n2);
                    if (ierr == 0)
                        i1 = i1 + n2;
                    else
                        i1 = i2;
                    ilst = i1;
                }
            }
            sorting_window_size = ilst;
        }

        // Recalculate the eigenvalues
        idx_t i = 0;
        while (i < jw)
        {
            idx_t n1 = 1;
            if (!is_complex<T>::value)
                if (i + 1 < jw)
                    if (TW(i + 1, i) != zero)
                        n1 = 2;

            if (n1 == 1)
                s[kwtop + i] = TW(i, i);
            else
                lahqr_eig22(TW(i, i), TW(i, i + 1), TW(i + 1, i), TW(i + 1, i + 1), s[kwtop + i], s[kwtop + i + 1]);
            i = i + n1;
        }

        // Reduce A back to Hessenberg form (if neccesary)
        if (s_spike != zero)
        {

            // Reflect spike back
            {
                T tau;
                auto v = slice(WV, pair{0, ns}, 0);
                for (idx_t i = 0; i < ns; ++i)
                {
                    v[i] = conj(V(0, i));
                }
                larfg(v, tau);
                
                auto Wv_aux = slice(WV, pair{0, jw}, 1);
                workspace_opts_t<> work2(Wv_aux);
                
                auto TW_slice = slice(TW, pair{0, ns}, pair{0, jw});
                larf(Side::Left, forward, v, conj(tau), TW_slice, work2);
                
                auto TW_slice2 = slice(TW, pair{0, jw}, pair{0, ns});
                larf(Side::Right, forward, v, tau, TW_slice2, work2);
                
                auto V_slice = slice(V, pair{0, jw}, pair{0, ns});
                larf(Side::Right, forward, v, tau, V_slice, work2);
            }

            // Hessenberg reduction
            {
                auto tau = slice(WV, pair{0, jw}, 0);
                gehrd(0, ns, TW, tau, gehrdOpts);

                workspace_opts_t<> work2(slice(WV, pair{0, jw}, 1));
                unmhr(Side::Right, Op::NoTrans, 0, ns, TW, tau, V, work2);
            }
        }

        // Copy the deflation window back into place
        if (kwtop > 0)
            A(kwtop, kwtop - 1) = s_spike * conj(V(0, 0));
        for (idx_t j = 0; j < jw; ++j)
            for (idx_t i = 0; i < std::min(j + 2, jw); ++i)
                A(kwtop + i, kwtop + j) = TW(i, j);

        // Store number of deflated eigenvalues
        nd = jw - ns;
        ns = ns - infqr;

        //
        // Update rest of the matrix using matrix matrix multiplication
        //
        idx_t istart_m, istop_m;
        if (want_t)
        {
            istart_m = 0;
            istop_m = n;
        }
        else
        {
            istart_m = ilo;
            istop_m = ihi;
        }
        // Horizontal multiply
        if (ihi < istop_m)
        {
            idx_t i = ihi;
            while (i < istop_m)
            {
                idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                auto A_slice = slice(A, pair{kwtop, ihi}, pair{i, i + iblock});
                auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::ConjTrans, Op::NoTrans, one, V, A_slice, WH_slice);
                lacpy(Uplo::General, WH_slice, A_slice);
                i = i + iblock;
            }
        }
        // Vertical multiply
        if (istart_m < kwtop)
        {
            idx_t i = istart_m;
            while (i < kwtop)
            {
                idx_t iblock = std::min<idx_t>(kwtop - i, nrows(WV));
                auto A_slice = slice(A, pair{i, i + iblock}, pair{kwtop, ihi});
                auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, A_slice, V, WV_slice);
                lacpy(Uplo::General, WV_slice, A_slice);
                i = i + iblock;
            }
        }
        // Update Z (also a vertical multiplication)
        if (want_z)
        {
            idx_t i = 0;
            while (i < n)
            {
                idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
                auto Z_slice = slice(Z, pair{i, i + iblock}, pair{kwtop, ihi});
                auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, V, WV_slice);
                lacpy(Uplo::General, WV_slice, Z_slice);
                i = i + iblock;
            }
        }
    }

    /** agressive_early_deflation accepts as input an upper Hessenberg matrix
     *  H and performs an orthogonal similarity transformation
     *  designed to detect and deflate fully converged eigenvalues from
     *  a trailing principal submatrix.  On output H has been over-
     *  written by a new Hessenberg matrix that is a perturbation of
     *  an orthogonal similarity transformation of H.  It is to be
     *  hoped that the final version of H has many zero subdiagonal
     *  entries.
     *
     * @param[in] want_t bool.
     *      If true, the full Schur factor T will be computed.
     *
     * @param[in] want_z bool.
     *      If true, the Schur vectors Z will be computed.
     *
     * @param[in] ilo    integer.
     *      Either ilo=0 or A(ilo,ilo-1) = 0.
     *
     * @param[in] ihi    integer.
     *      ilo and ihi determine an isolated block in A.
     *
     * @param[in] nw    integer.
     *      Desired window size to perform agressive early deflation on.
     *      If the matrix is not large enough to provide the scratch space
     *      or if the isolated block is small, a smaller value may be used.
     *
     * @param[in,out] A  n by n matrix.
     *       Hessenberg matrix on which AED will be performed
     *
     * @param[out] s  size n vector.
     *      On exit, the entries s[ihi-nd-ns:ihi-nd] contain the unconverged
     *      eigenvalues that can be used a shifts. The entries s[ihi-nd:ihi]
     *      contain the converged eigenvalues. Entries outside the range
     *      s[ihi-nw:ihi] are not changed. The converged shifts are stored
     *      in the same positions as their correspinding diagonal elements
     *      in A.
     *
     * @param[in,out] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors
     *      On exit, the orthogonal updates applied to A accumulated
     *      into Z.
     *
     * @param[out] ns    integer.
     *      Number of eigenvalues available as shifts in s.
     *
     * @param[out] nd    integer.
     *      Number of converged eigenvalues available as shifts in s.
     *
     * @ingroup computational
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<
            is_complex< type_t<vector_t> >::value
        , int > = 0
    >
    inline
    void agressive_early_deflation(
        bool want_t, 
        bool want_z, 
        size_type<matrix_t> ilo, 
        size_type<matrix_t> ihi, 
        size_type<matrix_t> nw, 
        matrix_t &A, 
        vector_t &s, 
        matrix_t &Z, 
        size_type<matrix_t> &ns, 
        size_type<matrix_t> &nd )
    {
        francis_opts_t< size_type<matrix_t> > opts = {};
        agressive_early_deflation( want_t, want_z, ilo, ihi, nw, A, s, Z, ns, nd, opts );
    }

} // lapack

#endif // TLAPACK_AED_HH
