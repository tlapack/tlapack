/// @file multishift_QR_sweep.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __QR_SWEEP_HH__
#define __QR_SWEEP_HH__

#include <memory>
#include <complex>

#include "legacy_api/blas/utils.hpp"
#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lahqr_shiftcolumn.hpp"

namespace lapack
{

    /** Given a 4-by-3 matrix H and small order reflector v,
     *  move_bulge applies the delayed right update to the last
     *  row and calculates a new reflector to move the bulge
     *  down. If the bulge collapses, an attempt is made to
     *  reintroduce it using shifts s1 and s2.
     *
     * @param[in] H 4x3 matrix.
     * @param[out] v vector of size 3
     *      On entry, the delayed reflector to apply
     *      The first element of the reflector is assumed to be one, and v[0] instead stores tau.
     *      On exit, the reflector that moves the bulge down one position
     * @param[in] s1 complex valued shift
     * @param[in] s2 complex valued shift
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        typename T = type_t<matrix_t>,
        typename real_t = real_type<T>>
    void move_bulge(matrix_t &H, vector_t &v, std::complex<real_t> s1, std::complex<real_t> s2)
    {

        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;
        using blas::abs1;
        using blas::conj;
        using blas::uroundoff;
        const T zero(0);
        const real_t eps = uroundoff<real_t>();

        // Perform delayed update of row below the bulge
        // Assumes the first two elements of the row are zero
        auto refsum = v[0] * v[2] * H(3, 2);
        H(3, 0) = -refsum;
        H(3, 1) = -refsum * conj(v[1]);
        H(3, 2) = H(3, 2) - refsum * conj(v[2]);

        // Generate reflector to move bulge down
        T tau, beta;
        v[0] = H(1, 0);
        v[1] = H(2, 0);
        v[2] = H(3, 0);
        larfg(v, tau);
        beta = v[0];
        v[0] = tau;

        // Check for bulge collapse
        if (H(3, 0) != zero or H(3, 1) != zero or H(3, 2) != zero)
        {
            // The bulge hasn't collapsed, typical case
            H(1, 0) = beta;
            H(2, 0) = zero;
            H(3, 0) = zero;
        }
        else
        {
            // The bulge has collapsed, attempt to reintroduce using
            // 2-small-subdiagonals trick
            std::unique_ptr<T[]> _vt(new T[3]);
            auto vt = legacyVector<T>(3, &_vt[0]);
            auto H2 = slice(H, pair{1, 4}, pair{0, 3});
            lahqr_shiftcolumn(H2, vt, s1, s2);
            larfg(vt, tau);
            vt[0] = tau;

            refsum = conj(v[0]) * H(1, 0) + conj(v[1]) * H(2, 0);
            if (abs1(H(2, 0) - refsum * vt[1]) + abs1(refsum * vt[2]) > eps * (abs1(H(0, 0)) + abs1(H(1, 1)) + abs1(H(2, 2))))
            {
                // Starting a new bulge here would create non-negligible fill. Use the old one.
                H(1, 0) = beta;
                H(2, 0) = zero;
                H(3, 0) = zero;
            }
            else
            {
                // Fill-in is negligible, use the new reflector.
                H(1, 0) = H(1, 0) - refsum;
                H(2, 0) = zero;
                H(3, 0) = zero;
                v[0] = vt[0];
                v[1] = vt[1];
                v[2] = vt[2];
            }
        }
    }

    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void multishift_QR_sweep(bool want_t, bool want_z, size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &s, matrix_t &Z)
    {

        using T = type_t<matrix_t>;
        using real_t = real_type<T>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;
        using blas::abs;
        using blas::abs1;
        using blas::conj;
        using blas::uroundoff;
        using blas::internal::colmajor_matrix;

        const real_t rzero(0);
        const T one(1);
        const T zero(0);
        const real_t eps = uroundoff<real_t>();
        const idx_t n = ncols(A);

        idx_t n_shifts = size(s);
        idx_t n_bulges = n_shifts / 2;

        const idx_t n_block_desired = n_shifts + 2;

        // Define workspace matrices
        std::unique_ptr<T[]> _V(new T[n_bulges * 3]);
        // V stores the delayed reflectors
        auto V = colmajor_matrix<T>(&_V[0], 3, n_bulges);

        std::unique_ptr<T[]> _U(new T[n_block_desired * n_block_desired]);
        // U stores the accumulated reflectors
        auto U = colmajor_matrix<T>(&_U[0], n_block_desired, n_block_desired);

        std::unique_ptr<T[]> _WH(new T[n_block_desired * n]);
        // WH is a workspace array used for the horizontal multiplications
        auto WH = colmajor_matrix<T>(&_WH[0], n_block_desired, n);

        std::unique_ptr<T[]> _WV(new T[n_block_desired * n]);
        // WH is a workspace array used for the vertical multiplications
        auto WV = colmajor_matrix<T>(&_WV[0], n, n_block_desired);

        // i_pos_block points to the start of the block of bulges
        idx_t i_pos_block;

        //
        // The following code block introduces the bulges into the matrix
        //
        {

            // Near-the-diagonal bulge introduction
            // The calculations are initially limited to the window: A(ilo:ilo+n_block,ilo:ilo+n_block)
            // The rest is updated later via level 3 BLAS
            idx_t n_block = n_block_desired;
            idx_t istart_m = ilo;
            idx_t istop_m = ilo + n_block;
            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                for (idx_t i_pos = ilo; i_pos < ilo + n_block - 2 * (i_bulge + 1); ++i_pos)
                {
                    auto v = col(V, i_bulge);
                    if (i_pos == ilo)
                    {
                        // Introduce bulge
                        T tau;
                        auto H = slice(A, pair{ilo, ilo + 3}, pair{ilo, ilo + 3});
                        lahqr_shiftcolumn(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);
                        larfg(v, tau);
                        v[0] = tau;
                    }
                    else
                    {
                        // Chase bulge down
                        auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 2});
                        move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);
                    }

                    auto t1 = conj(v[0]);
                    auto v2 = v[1];
                    auto t2 = t1 * v2;
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];
                    // Apply the reflector we just calculated from the right (but leave the last row for later)
                    for (idx_t j = istart_m; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                    }

                    //
                    // TODO: the following two loops can be moved outside this loop (which is a bit more efficient)
                    //

                    // Apply the reflector we just calculated from the left
                    for (idx_t j = i_pos; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * t1;
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                    }
                    // Accumulate the reflector into U
                    // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                    for (idx_t j = 0; j < nrows(U2); ++j)
                    {
                        auto sum = U2(j, i_pos - ilo) + v2 * U2(j, i_pos - ilo + 1) + v3 * U2(j, i_pos - ilo + 2);
                        U2(j, i_pos - ilo) = U2(j, i_pos - ilo) - sum * conj(t1);
                        U2(j, i_pos - ilo + 1) = U2(j, i_pos - ilo + 1) - sum * conj(t2);
                        U2(j, i_pos - ilo + 2) = U2(j, i_pos - ilo + 2) - sum * conj(t3);
                    }
                }
            }
            // Update rest of the matrix
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
            if (ilo + n_shifts + 1 < istop_m)
            {
                auto A_slice = slice(A, pair{ilo, ilo + n_block}, pair{ilo + n_block, istop_m});
                auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                lacpy(Uplo::General, WH_slice, A_slice);
            }
            // Vertical multiply
            if (istart_m < ilo)
            {
                auto A_slice = slice(A, pair{istart_m, ilo}, pair{ilo, ilo + n_block});
                auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, A_slice);
            }
            // Update Z (also a vertical multiplication)
            if (want_z)
            {
                auto Z_slice = slice(Z, pair{0, n}, pair{ilo, ilo + n_block});
                auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, Z_slice);
            }

            i_pos_block = n_block - n_shifts;
        }

        //
        // The following code block moves the bulges down untill they are low enough to be removed
        //
        while (i_pos_block < ihi - n_block_desired)
        {

            // Number of positions each bulge will be moved down
            idx_t n_pos = std::min(n_block_desired - n_shifts, ihi - n_shifts - 1 - i_pos_block);
            // Actual blocksize
            idx_t n_block = n_shifts + n_pos;

            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            // Near-the-diagonal bulge chase
            // The calculations are initially limited to the window: A(i_pos_block-1:i_pos_block+n_block,i_pos_block:i_pos_block+n_block)
            // The rest is updated later via level 3 BLAS

            idx_t istart_m = i_pos_block;
            idx_t istop_m = i_pos_block + n_block;
            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                for (idx_t i_pos = i_pos_block + 2 * (n_bulges - i_bulge - 1); i_pos < i_pos_block + 2 * (n_bulges - i_bulge - 1) + n_pos; ++i_pos)
                {
                    auto v = col(V, i_bulge);
                    auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 2});
                    move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                    auto t1 = conj(v[0]);
                    auto v2 = v[1];
                    auto t2 = t1 * v2;
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];
                    // Apply the reflector we just calculated from the right (but leave the last row for later)
                    for (idx_t j = istart_m; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                    }

                    //
                    // TODO: the following two loops can be moved outside this loop (which is a bit more efficient)
                    //

                    // Apply the reflector we just calculated from the left
                    for (idx_t j = i_pos; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * t1;
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                    }
                    // Accumulate the reflector into U
                    // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                    for (idx_t j = 0; j < nrows(U2); ++j)
                    {
                        auto sum = U2(j, i_pos - i_pos_block) + v2 * U2(j, i_pos - i_pos_block + 1) + v3 * U2(j, i_pos - i_pos_block + 2);
                        U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * conj(t1);
                        U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * conj(t2);
                        U2(j, i_pos - i_pos_block + 2) = U2(j, i_pos - i_pos_block + 2) - sum * conj(t3);
                    }
                }
            }
            // Update rest of the matrix
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
            if (i_pos_block + n_block < istop_m)
            {
                auto A_slice = slice(A, pair{i_pos_block, i_pos_block + n_block}, pair{i_pos_block + n_block, istop_m});
                auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                lacpy(Uplo::General, WH_slice, A_slice);
            }
            // Vertical multiply
            if (istart_m < i_pos_block)
            {
                auto A_slice = slice(A, pair{istart_m, i_pos_block}, pair{i_pos_block, i_pos_block + n_block});
                auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, A_slice);
            }
            // Update Z (also a vertical multiplication)
            if (want_z)
            {
                auto Z_slice = slice(Z, pair{0, n}, pair{i_pos_block, i_pos_block + n_block});
                auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, Z_slice);
            }

            i_pos_block = i_pos_block + n_pos;
        }

        //
        // The following code removes the bulges from the matrix
        //
        {
            idx_t n_block = ihi-i_pos_block;

            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            // Near-the-diagonal bulge chase
            // The calculations are initially limited to the window: A(i_pos_block-1:ihi,i_pos_block:ihi)
            // The rest is updated later via level 3 BLAS

            idx_t istart_m = i_pos_block;
            idx_t istop_m = ihi;
            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                for (idx_t i_pos = i_pos_block + 2 * (n_bulges - i_bulge - 1); i_pos < ihi-1; ++i_pos)
                {
                    if( i_pos == ihi-2 ){
                        // Special case, the bulge is at the bottom, needs a smaller reflector (order 2)
                        auto v = slice(V, pair{0,2}, i_bulge);
                        auto h = slice(A, pair{i_pos, i_pos + 2}, i_pos - 1);
                        larfg( h, v[0]);
                        v[1] = h[1];
                        h[1] = zero;

                        auto t1 = conj(v[0]);
                        auto v2 = v[1];
                        auto t2 = t1 * v2;
                        // Apply the reflector we just calculated from the right
                        for (idx_t j = istart_m; j < i_pos + 2; ++j)
                        {
                            auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1);
                            A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        }

                        //
                        // TODO: the following two loops can be moved outside this loop (which is a bit more efficient)
                        //

                        // Apply the reflector we just calculated from the left
                        for (idx_t j = i_pos; j < istop_m; ++j)
                        {
                            auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j);
                            A(i_pos, j) = A(i_pos, j) - sum * t1;
                            A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        }
                        // Accumulate the reflector into U
                        // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                        for (idx_t j = 0; j < nrows(U2); ++j)
                        {
                            auto sum = U2(j, i_pos - i_pos_block) + v2 * U2(j, i_pos - i_pos_block + 1);
                            U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * conj(t1);
                            U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * conj(t2);
                        }

                    } else {
                        auto v = col(V, i_bulge);
                        auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 2});
                        move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                        auto t1 = conj(v[0]);
                        auto v2 = v[1];
                        auto t2 = t1 * v2;
                        auto v3 = v[2];
                        auto t3 = t1 * v[2];
                        // Apply the reflector we just calculated from the right (but leave the last row for later)
                        for (idx_t j = istart_m; j < i_pos + 3; ++j)
                        {
                            auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                            A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                            A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                        }

                        //
                        // TODO: the following two loops can be moved outside this loop (which is a bit more efficient)
                        //

                        // Apply the reflector we just calculated from the left
                        for (idx_t j = i_pos; j < istop_m; ++j)
                        {
                            auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                            A(i_pos, j) = A(i_pos, j) - sum * t1;
                            A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                            A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                        }
                        // Accumulate the reflector into U
                        // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                        for (idx_t j = 0; j < nrows(U2); ++j)
                        {
                            auto sum = U2(j, i_pos - i_pos_block) + v2 * U2(j, i_pos - i_pos_block + 1) + v3 * U2(j, i_pos - i_pos_block + 2);
                            U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * conj(t1);
                            U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * conj(t2);
                            U2(j, i_pos - i_pos_block + 2) = U2(j, i_pos - i_pos_block + 2) - sum * conj(t3);
                        }
                    }
                }
            }
            // Update rest of the matrix
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
                auto A_slice = slice(A, pair{i_pos_block, ihi}, pair{ihi, istop_m});
                auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                lacpy(Uplo::General, WH_slice, A_slice);
            }
            // Vertical multiply
            if (istart_m < i_pos_block)
            {
                auto A_slice = slice(A, pair{istart_m, i_pos_block}, pair{i_pos_block, ihi});
                auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, A_slice);
            }
            // Update Z (also a vertical multiplication)
            if (want_z)
            {
                auto Z_slice = slice(Z, pair{0, n}, pair{i_pos_block, ihi});
                auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                lacpy(Uplo::General, WV_slice, Z_slice);
            }
        }

    }

} // lapack

#endif // __QR_SWEEP_HH__
