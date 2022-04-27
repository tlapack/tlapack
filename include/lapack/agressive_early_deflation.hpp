/// @file agressive_early_deflation.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __AED_HH__
#define __AED_HH__

#include <memory>
#include <complex>

#include "legacy_api/blas/utils.hpp"
#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/larf.hpp"
#include "lapack/lahqr.hpp"
#include "lapack/gehd2.hpp"
#include "lapack/unghr.hpp"

namespace lapack
{

    template <
        class matrix_t,
        class vector_t,
        typename idx_t = size_type<matrix_t>,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void agressive_early_deflation(bool want_t, bool want_z, idx_t ilo, idx_t ihi, idx_t nw, matrix_t &A, vector_t &s, matrix_t &Z)
    {

        using T = type_t<matrix_t>;
        using real_t = real_type<T>;
        using pair = std::pair<idx_t, idx_t>;
        using blas::abs;
        using blas::abs1;
        using blas::conj;
        using blas::uroundoff;
        using std::max;

        using blas::internal::colmajor_matrix;

        const real_t rzero(0);
        const T one(1);
        const T zero(0);
        const idx_t n = ncols(A);
        const real_t eps = uroundoff<real_t>();
        const real_t small_num = blas::safe_min<real_t>() * ((T)n / blas::uroundoff<real_t>());
        // Size of the deflation window
        const idx_t jw = std::min(nw, ihi - ilo);
        // First row index in the deflation window
        const idx_t kwtop = ihi - jw;

        // Define workspace matrices
        std::unique_ptr<T[]> _V(new T[jw * jw]);
        // V stores the orthogonal transformations
        auto V = colmajor_matrix<T>(&_V[0], jw, jw);

        std::unique_ptr<T[]> _TW(new T[jw * jw]);
        // TW stores a copy of the deflation window
        auto TW = colmajor_matrix<T>(&_TW[0], jw, jw);

        std::unique_ptr<T[]> _WV(new T[jw * n]);
        // WH is a workspace array used for the vertical multiplications
        auto WV = colmajor_matrix<T>(&_WV[0], n, jw);

        // WH is a workspace array used for the horizontal multiplications
        // This can reuse the WH space in memory, because we will never use it at the same time
        auto WH = colmajor_matrix<T>(&_WV[0], jw, n);

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

            // TODO
        }

        // Convert the window to spike-triangular form. i.e. calculate the
        // Schur form of the deflation window.
        // If the QR algorithm fails to convergence, it can still be
        // partially in Schur form. In that case we continue on a smaller
        // number of eigenvalues.
        auto A_window = slice(A, pair{kwtop, ihi}, pair{kwtop, ihi});
        auto s_window = slice(s, pair{kwtop, ihi});
        lacpy(Uplo::General, A_window, TW);
        laset(Uplo::General, zero, one, V);
        int infqr = lahqr(true, true, 0, jw, TW, s_window, V);

        // Deflation detection loop
        idx_t ns = jw;
        idx_t ilst = infqr + 1;
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
                auto foo = abs1(TW(ns - 1, ns - 1));
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
                auto foo = abs(TW(ns - 1, ns - 1)) + sqrt(abs(TW(ns - 1, ns - 2))) * sqrt(abs(TW(ns - 2, ns - 1)));
                if (foo == zero)
                    foo = abs(s_spike);
                if (max(abs(s_spike * V(0, ns - 1)), abs(s_spike * V(0, ns - 2))) <= max<real_t>(small_num, eps * foo))
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

        // Number of unconverges eigenvalues available as shifts

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
                auto work2 = slice(WV, pair{0, jw}, 1);
                auto TW_slice = slice(TW, pair{0, ns}, pair{0, jw});
                larf(Side::Left, v, tau, TW_slice, work2);
                TW_slice = slice(TW, pair{0, jw}, pair{0, ns});
                larf(Side::Right, v, tau, TW_slice, work2);
                auto V_slice = slice(V, pair{0, jw}, pair{0, ns});
                larf(Side::Right, v, tau, V_slice, work2);
            }

            // Hessenberg reduction
            {
                auto tau = slice(WV, pair{0, jw}, 0);
                auto work2 = slice(WV, pair{0, jw}, 1);
                gehd2(0, ns, TW, tau, work2);
                unmhr(Side::Right, Op::NoTrans, 0, ns, TW, tau, V, work2);
            }
        }

        // Copy the deflation window back into place
        if(kwtop > 0)
            A(kwtop, kwtop - 1) = s_spike * conj(V(0,0));
        for (idx_t j = 0; j < jw; ++j)
            for (idx_t i = 0; i < std::min(j + 2, jw); ++i)
                A(kwtop + i, kwtop + j) = TW(i, j);

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
            auto A_slice = slice(A, pair{kwtop, ihi}, pair{ihi, istop_m});
            auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
            gemm(Op::ConjTrans, Op::NoTrans, one, V, A_slice, zero, WH_slice);
            lacpy(Uplo::General, WH_slice, A_slice);
        }
        // Vertical multiply
        if (istart_m < kwtop)
        {
            auto A_slice = slice(A, pair{istart_m, kwtop}, pair{kwtop, ihi});
            auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
            gemm(Op::NoTrans, Op::NoTrans, one, A_slice, V, zero, WV_slice);
            lacpy(Uplo::General, WV_slice, A_slice);
        }
        // Update Z (also a vertical multiplication)
        if (want_z)
        {
            auto Z_slice = slice(Z, pair{0, n}, pair{kwtop, ihi});
            auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
            gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, V, zero, WV_slice);
            lacpy(Uplo::General, WV_slice, Z_slice);
        }
    }

} // lapack

#endif // __AED_HH__
