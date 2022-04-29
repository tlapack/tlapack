/// @file multishift_QR_sweep_rowoptimized.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __QR_SWEEP_ROW_HH__
#define __QR_SWEEP_ROW_HH__

#include <memory>
#include <complex>

#include "legacy_api/blas/utils.hpp"
#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lahqr_shiftcolumn.hpp"
#include "lapack/move_bulge.hpp"

//
// This function is a work in progress, the main focus is on the column major routine for now,
// But it would be a waste to throw this away.
// 


namespace lapack
{

    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true,
        enable_if_t<is_same_v< layout_type<matrix_t>, RowMajor_t >, bool> = true>
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

        // We should really be using rowmajor matrix for this routine, as it is optimized for row major matrices
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
        auto V = colmajor_matrix<T>(&_V[0], n_bulges, 3);

        std::unique_ptr<T[]> _U(new T[n_block_desired * n_block_desired]);
        // U stores the accumulated reflectors
        auto U = colmajor_matrix<T>(&_U[0], n_block_desired, n_block_desired);

        std::unique_ptr<T[]> _WH(new T[n_block_desired * n]);
        // WH is a workspace array used for the horizontal multiplications
        auto WH = colmajor_matrix<T>(&_WH[0], n_block_desired, n);

        // WH is a workspace array used for the vertical multiplications
        // This can reuse the WH space in memory, because we will never use it at the same time
        auto WV = colmajor_matrix<T>(&_WH[0], n, n_block_desired);

        // V2 stores a sequence of reflectors
        // This can reuse the WH space in memory, because we will never use it at the same time
        auto V_block = colmajor_matrix<T>(&_WH[0], n_block_desired, 3);

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
                    auto v = row(V, i_bulge);
                    auto v_block = row(V_block, i_pos - ilo);
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
                    v_block[0] = v[0];
                    v_block[1] = v[1];
                    v_block[2] = v[2];

                    auto t1 = conj(v[0]);
                    auto v2 = v[1];
                    auto t2 = t1 * v2;
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];


                    // Apply the reflector we just calculated from the left
                    for (idx_t j = i_pos; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * t1;
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                    }

                    // Apply the reflector we just calculated from the right
                    // We leave the last row for later (it interferes with the optimally packed bulges)
                    // The first rows are also left for later (for efficiency)
                    // We only end up updating 2 rows
                    for (idx_t j = i_pos + 1; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                    }
                }

                // Perform the delayed updates from the right (but not the last row)
                for (idx_t i_pos = ilo; i_pos < ilo + n_block - 2 * (i_bulge + 1); ++i_pos)
                {
                    for (idx_t j = istart_m; j < i_pos + 1; ++j)
                    {
                        auto v = row(V_block, i_pos - ilo);
                        auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * v[0];
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                    }
                }
                // Accumulate the reflectors into U
                // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                for (idx_t i_pos = ilo; i_pos < ilo + n_block - 2 * (i_bulge + 1); ++i_pos)
                {
                    for (idx_t j = 0; j < nrows(U2); ++j)
                    {
                        auto v = row(V_block, i_pos - ilo);
                        auto sum = U2(j, i_pos - ilo) + v[1] * U2(j, i_pos - ilo + 1) + v[2] * U2(j, i_pos - ilo + 2);
                        U2(j, i_pos - ilo) = U2(j, i_pos - ilo) - sum * v[0];
                        U2(j, i_pos - ilo + 1) = U2(j, i_pos - ilo + 1) - sum * v[0] * conj(v[1]);
                        U2(j, i_pos - ilo + 2) = U2(j, i_pos - ilo + 2) - sum * v[0] * conj(v[2]);
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
                idx_t i_pos_start = i_pos_block + 2 * (n_bulges - i_bulge - 1);
                for (idx_t i_pos = i_pos_start; i_pos < i_pos_start + n_pos; ++i_pos)
                {
                    auto v = row(V, i_bulge);
                    auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 2});
                    move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                    // Store reflector for later
                    auto v_block = row(V_block, i_pos - i_pos_block);
                    v_block[0] = v[0];
                    v_block[1] = v[1];
                    v_block[2] = v[2];

                    auto t1 = conj(v[0]);
                    auto v2 = v[1];
                    auto t2 = t1 * v2;
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];

                    // Apply the reflector we just calculated from the left
                    for (idx_t j = i_pos; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * t1;
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                    }

                    // Apply the reflector we just calculated from the right
                    // We leave the last row for later (it interferes with the optimally packed bulges)
                    // The first rows are also left for later (for efficiency)
                    // We only end up updating 2 rows
                    for (idx_t j = i_pos + 1; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                    }

                }

                // Perform the delayed updates from the right (but not the last row)
                for (idx_t i_pos = i_pos_start; i_pos < i_pos_start + n_pos; ++i_pos)
                {
                    for (idx_t j = istart_m; j < i_pos + 1; ++j)
                    {
                        auto v = row(V_block, i_pos - i_pos_block);
                        auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * v[0];
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                    }
                }
                // Accumulate the reflectors into U
                // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                for (idx_t i_pos = i_pos_start; i_pos < i_pos_start + n_pos; ++i_pos)
                {
                    for (idx_t j = 0; j < nrows(U2); ++j)
                    {
                        auto v = row(V_block, i_pos - i_pos_block);
                        auto sum = U2(j, i_pos - i_pos_block) + v[1] * U2(j, i_pos - i_pos_block + 1) + v[2] * U2(j, i_pos - i_pos_block + 2);
                        U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * v[0];
                        U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * v[0] * conj(v[1]);
                        U2(j, i_pos - i_pos_block + 2) = U2(j, i_pos - i_pos_block + 2) - sum * v[0] * conj(v[2]);
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
            idx_t n_block = ihi - i_pos_block;

            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            // Near-the-diagonal bulge chase
            // The calculations are initially limited to the window: A(i_pos_block-1:ihi,i_pos_block:ihi)
            // The rest is updated later via level 3 BLAS

            idx_t istart_m = i_pos_block;
            idx_t istop_m = ihi;
            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                idx_t i_pos_start = i_pos_block + 2 * (n_bulges - i_bulge - 1);
                for (idx_t i_pos = i_pos_start; i_pos < ihi - 1; ++i_pos)
                {
                    if (i_pos == ihi - 2)
                    {
                        // Special case, the bulge is at the bottom, needs a smaller reflector (order 2)
                        auto v = slice(V, i_bulge, pair{0, 2} );
                        auto h = slice(A, pair{i_pos, i_pos + 2}, i_pos - 1);
                        larfg(h, v[0]);
                        v[1] = h[1];
                        h[1] = zero;

                        // Store reflector for later
                        auto v_block = row(V_block, i_pos - i_pos_block);
                        v_block[0] = v[0];
                        v_block[1] = v[1];
                        v_block[2] = zero;

                        auto t1 = conj(v[0]);
                        auto v2 = v[1];
                        auto t2 = t1 * v2;

                        // Apply the reflector we just calculated from the left
                        for (idx_t j = i_pos; j < istop_m; ++j)
                        {
                            auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j);
                            A(i_pos, j) = A(i_pos, j) - sum * t1;
                            A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                        }

                        // Apply the reflector we just calculated from the right
                        // We leave the last row for later (it interferes with the optimally packed bulges)
                        // The first rows are also left for later (for efficiency)
                        // We only end up updating 1 row
                        for (idx_t j = i_pos + 1; j < i_pos + 2; ++j)
                        {
                            auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1);
                            A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                        }
                    }
                    else
                    {
                        auto v = row(V, i_bulge);
                        auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 2});
                        move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                        // Store reflector for later
                        auto v_block = row(V_block, i_pos - i_pos_block);
                        v_block[0] = v[0];
                        v_block[1] = v[1];
                        v_block[2] = v[2];

                        auto t1 = conj(v[0]);
                        auto v2 = v[1];
                        auto t2 = t1 * v2;
                        auto v3 = v[2];
                        auto t3 = t1 * v[2];

                        // Apply the reflector we just calculated from the left
                        for (idx_t j = i_pos; j < istop_m; ++j)
                        {
                            auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j) + conj(v3) * A(i_pos + 2, j);
                            A(i_pos, j) = A(i_pos, j) - sum * t1;
                            A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                            A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t3;
                        }

                        // Apply the reflector we just calculated from the right
                        // We leave the last row for later (it interferes with the optimally packed bulges)
                        // The first rows are also left for later (for efficiency)
                        // We only end up updating 2 rows
                        for (idx_t j = i_pos + 1; j < i_pos + 3; ++j)
                        {
                            auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1) + v3 * A(j, i_pos + 2);
                            A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                            A(j, i_pos + 2) = A(j, i_pos + 2) - sum * conj(t3);
                        }

                    }
                }

                // Perform the delayed updates from the right (but not the last row)
                for (idx_t i_pos = i_pos_start; i_pos < ihi-1; ++i_pos)
                {
                    if(i_pos == ihi-2){

                        for (idx_t j = istart_m; j < i_pos + 1; ++j)
                        {
                            auto v = row(V_block, i_pos - i_pos_block);
                            auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1);
                            A(j, i_pos) = A(j, i_pos) - sum * v[0];
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                        }
                    } else{
                        for (idx_t j = istart_m; j < i_pos + 1; ++j)
                        {
                            auto v = row(V_block, i_pos - i_pos_block);
                            auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                            A(j, i_pos) = A(j, i_pos) - sum * v[0];
                            A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                            A(j, i_pos + 2) = A(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                        }
                    }
                }

                // Accumulate the reflectors into U
                // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                for (idx_t i_pos = i_pos_start; i_pos < ihi-1; ++i_pos)
                {
                    if(i_pos == ihi-2){

                        for (idx_t j = 0; j < nrows(U2); ++j)
                        {
                            auto v = row(V_block, i_pos - i_pos_block);
                            auto sum = U2(j, i_pos - i_pos_block) + v[1] * U2(j, i_pos - i_pos_block + 1);
                            U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * v[0];
                            U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * v[0] * conj(v[1]);
                        }
                    } else{
                        for (idx_t j = 0; j < nrows(U2); ++j)
                        {
                            auto v = row(V_block, i_pos - i_pos_block);
                            auto sum = U2(j, i_pos - i_pos_block) + v[1] * U2(j, i_pos - i_pos_block + 1) + v[2] * U2(j, i_pos - i_pos_block + 2);
                            U2(j, i_pos - i_pos_block) = U2(j, i_pos - i_pos_block) - sum * v[0];
                            U2(j, i_pos - i_pos_block + 1) = U2(j, i_pos - i_pos_block + 1) - sum * v[0] * conj(v[1]);
                            U2(j, i_pos - i_pos_block + 2) = U2(j, i_pos - i_pos_block + 2) - sum * v[0] * conj(v[2]);
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

#endif // __QR_SWEEP_ROW_HH__
