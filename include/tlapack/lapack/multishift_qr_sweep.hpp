/// @file multishift_QR_sweep_coloptimized.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_QR_SWEEP_HH
#define TLAPACK_QR_SWEEP_HH

#include <memory>
#include <complex>

#include "tlapack/legacy_api/base/utils.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/lahqr_shiftcolumn.hpp"
#include "tlapack/lapack/move_bulge.hpp"

namespace tlapack
{
    /** multishift_QR_sweep performs a single small-bulge multi-shift QR sweep.
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
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix on which AED will be performed
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[in,out] Z  n by n matrix.
     *      On entry, the previously calculated Schur factors
     *      On exit, the orthogonal updates applied to A accumulated
     *      into Z.
     *
     * @param[out] V    3 by size(s)/2 matrix.
     *      Workspace matrix
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void multishift_QR_sweep(bool want_t, bool want_z, size_type<matrix_t> ilo, size_type<matrix_t> ihi, matrix_t &A, vector_t &s, matrix_t &Z, matrix_t &V)
    {

        using T = type_t<matrix_t>;
        using real_t = real_type<T>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;

        const T one(1);
        const T zero(0);
        const idx_t n = ncols(A);
        const real_t eps = ulp<real_t>();
        const real_t small_num = safe_min<real_t>() * ((real_t)n / eps);

        // Assertions
        assert(n >= 12);
        assert(nrows(A) == n);
        assert(ncols(Z) == n);
        assert(nrows(Z) == n);
        assert(nrows(V) == 3);
        assert(ncols(V) >= size(s) / 2);

        const idx_t n_block_max = (n - 3) / 3;
        const idx_t n_shifts_max = std::min(ihi - ilo - 1, std::max<idx_t>(2, 3 * (n_block_max / 4)));

        idx_t n_shifts = std::min<idx_t>(size(s), n_shifts_max);
        if (n_shifts % 2 == 1)
            n_shifts = n_shifts - 1;
        idx_t n_bulges = n_shifts / 2;

        const idx_t n_block_desired = std::min<idx_t>(2 * n_shifts, n_block_max);

        // Define workspace matrices
        // We use the lower triangular part of A as workspace

        // U stores the orthogonal transformations
        auto U = slice(A, pair{n - n_block_desired, n}, pair{0, n_block_desired});

        // Workspace for horizontal multiplications
        auto WH = slice(A, pair{n - n_block_desired, n}, pair{n_block_desired, n - n_block_desired - 3});

        // Workspace for vertical multiplications
        auto WV = slice(A, pair{n_block_desired + 3, n - n_block_desired}, pair{0, n_block_desired});

        // i_pos_block points to the start of the block of bulges
        idx_t i_pos_block;

        //
        // The following code block introduces the bulges into the matrix
        //
        {
            // Near-the-diagonal bulge introduction
            // The calculations are initially limited to the window: A(ilo:ilo+n_block,ilo:ilo+n_block)
            // The rest is updated later via level 3 BLAS
            idx_t n_block = std::min(n_block_desired, ihi - ilo);
            idx_t istart_m = ilo;
            idx_t istop_m = ilo + n_block;
            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            for (idx_t i_pos_last = ilo; i_pos_last < ilo + n_block - 2; ++i_pos_last)
            {
                // The number of bulges that are in the pencil
                idx_t n_active_bulges = std::min(n_bulges, ((i_pos_last - ilo) / 2) + 1);
                for (idx_t i_bulge = 0; i_bulge < n_active_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    if (i_pos == ilo)
                    {
                        // Introduce bulge
                        T tau;
                        auto H = slice(A, pair{ilo, ilo + 3}, pair{ilo, ilo + 3});
                        lahqr_shiftcolumn(H, v, s[size(s)- 1 - 2 * i_bulge], s[size(s)- 1 - 2 * i_bulge - 1]);
                        larfg(v, tau);
                        v[0] = tau;
                    }
                    else
                    {
                        // Chase bulge down
                        auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                        move_bulge(H, v, s[size(s)- 1 - 2 * i_bulge], s[size(s)- 1 - 2 * i_bulge - 1]);
                    }

                    // Apply the reflector we just calculated from the right
                    // We leave the last row for later (it interferes with the optimally packed bulges)
                    for (idx_t j = istart_m; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * v[0];
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                    }

                    // Apply the reflector we just calculated from the left
                    // We only update a single column, the rest is updated later
                    auto sum = A(i_pos, i_pos) + conj(v[1]) * A(i_pos + 1, i_pos) + conj(v[2]) * A(i_pos + 2, i_pos);
                    A(i_pos, i_pos) = A(i_pos, i_pos) - sum * conj(v[0]);
                    A(i_pos + 1, i_pos) = A(i_pos + 1, i_pos) - sum * conj(v[0]) * v[1];
                    A(i_pos + 2, i_pos) = A(i_pos + 2, i_pos) - sum * conj(v[0]) * v[2];

                    // Test for deflation.
                    if (i_pos > ilo)
                    {
                        if (A(i_pos, i_pos - 1) != zero)
                        {
                            auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                            if (tst1 == zero)
                            {
                                if (i_pos > ilo + 1)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 2));
                                if (i_pos > ilo + 2)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 3));
                                if (i_pos > ilo + 3)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 4));
                                if (i_pos < ihi - 1)
                                    tst1 += abs1(A(i_pos + 1, i_pos));
                                if (i_pos < ihi - 2)
                                    tst1 += abs1(A(i_pos + 2, i_pos));
                                if (i_pos < ihi - 3)
                                    tst1 += abs1(A(i_pos + 3, i_pos));
                            }
                            if (abs1(A(i_pos, i_pos - 1)) < std::max(small_num, eps * tst1))
                            {
                                auto ab = std::max(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto ba = std::min(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto aa = std::max(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto bb = std::min(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto s = aa + ab;
                                if (ba * (ab / s) <= std::max(small_num, eps * (bb * (aa / s))))
                                {
                                    A(i_pos, i_pos - 1) = zero;
                                }
                            }
                        }
                    }
                }

                // The following code performs the delayed update from the left
                // it is optimized for column oriented matrices, but the increased complexity
                // likely causes slower code
                // for (idx_t j = ilo; j < istop_m; ++j)
                // {
                //     idx_t i_bulge_start = (i_pos_last + 2 > j) ? (i_pos_last + 2 - j) / 2 : 0;
                //     for (idx_t i_bulge = i_bulge_start; i_bulge < n_active_bulges; ++i_bulge)
                //     {
                //         idx_t i_pos = i_pos_last - 2 * i_bulge;
                //         auto v = col(V, i_bulge);
                //         auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                //         A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                //         A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                //         A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                //     }
                // }

                // Delayed update from the left
                for (idx_t i_bulge = 0; i_bulge < n_active_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    for (idx_t j = i_pos + 1; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                    }
                }

                // Accumulate the reflectors into U
                for (idx_t i_bulge = 0; i_bulge < n_active_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    idx_t i1 = 0;
                    idx_t i2 = std::min(nrows(U2), (i_pos_last - ilo) + (i_pos_last - ilo) + 3);
                    for (idx_t j = i1; j < i2; ++j)
                    {
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
                idx_t i = ilo + n_block;
                while (i < istop_m)
                {
                    idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                    auto A_slice = slice(A, pair{ilo, ilo + n_block}, pair{i, i + iblock});
                    auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                    lacpy(Uplo::General, WH_slice, A_slice);
                    i = i + iblock;
                }
            }
            // Vertical multiply
            if (istart_m < ilo)
            {
                idx_t i = istart_m;
                while (i < ilo)
                {
                    idx_t iblock = std::min<idx_t>(ilo - i, nrows(WV));
                    auto A_slice = slice(A, pair{i, i + iblock}, pair{ilo, ilo + n_block});
                    auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
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
                    auto Z_slice = slice(Z, pair{i, i + iblock}, pair{ilo, ilo + n_block});
                    auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                    lacpy(Uplo::General, WV_slice, Z_slice);
                    i = i + iblock;
                }
            }

            i_pos_block = ilo + n_block - n_shifts;
        }

        //
        // The following code block moves the bulges down untill they are low enough to be removed
        //
        while (i_pos_block + n_block_desired < ihi)
        {
            // Number of positions each bulge will be moved down
            idx_t n_pos = std::min<idx_t>(n_block_desired - n_shifts, ihi - n_shifts - 1 - i_pos_block);
            // Actual blocksize
            idx_t n_block = n_shifts + n_pos;

            auto U2 = slice(U, pair{0, n_block}, pair{0, n_block});
            laset(Uplo::General, zero, one, U2);

            // Near-the-diagonal bulge chase
            // The calculations are initially limited to the window: A(i_pos_block-1:i_pos_block+n_block,i_pos_block:i_pos_block+n_block)
            // The rest is updated later via level 3 BLAS

            idx_t istart_m = i_pos_block;
            idx_t istop_m = i_pos_block + n_block;
            for (idx_t i_pos_last = i_pos_block + n_shifts - 2; i_pos_last < i_pos_block + n_shifts - 2 + n_pos; ++i_pos_last)
            {
                for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                    move_bulge(H, v, s[size(s)- 1 - 2 * i_bulge], s[size(s)- 1 - 2 * i_bulge - 1]);

                    // Apply the reflector we just calculated from the right
                    // We leave the last row for later (it interferes with the optimally packed bulges)
                    for (idx_t j = istart_m; j < i_pos + 3; ++j)
                    {
                        auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                        A(j, i_pos) = A(j, i_pos) - sum * v[0];
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                        A(j, i_pos + 2) = A(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                    }

                    // Apply the reflector we just calculated from the left
                    // We only update a single column, the rest is updated later
                    auto sum = A(i_pos, i_pos) + conj(v[1]) * A(i_pos + 1, i_pos) + conj(v[2]) * A(i_pos + 2, i_pos);
                    A(i_pos, i_pos) = A(i_pos, i_pos) - sum * conj(v[0]);
                    A(i_pos + 1, i_pos) = A(i_pos + 1, i_pos) - sum * conj(v[0]) * v[1];
                    A(i_pos + 2, i_pos) = A(i_pos + 2, i_pos) - sum * conj(v[0]) * v[2];

                    // Test for deflation.
                    if (i_pos > ilo)
                    {
                        if (A(i_pos, i_pos - 1) != zero)
                        {
                            auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                            if (tst1 == zero)
                            {
                                if (i_pos > ilo + 1)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 2));
                                if (i_pos > ilo + 2)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 3));
                                if (i_pos > ilo + 3)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 4));
                                if (i_pos < ihi - 1)
                                    tst1 += abs1(A(i_pos + 1, i_pos));
                                if (i_pos < ihi - 2)
                                    tst1 += abs1(A(i_pos + 2, i_pos));
                                if (i_pos < ihi - 3)
                                    tst1 += abs1(A(i_pos + 3, i_pos));
                            }
                            if (abs1(A(i_pos, i_pos - 1)) < std::max(small_num, eps * tst1))
                            {
                                auto ab = std::max(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto ba = std::min(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto aa = std::max(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto bb = std::min(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto s = aa + ab;
                                if (ba * (ab / s) <= std::max(small_num, eps * (bb * (aa / s))))
                                {
                                    A(i_pos, i_pos - 1) = zero;
                                }
                            }
                        }
                    }
                }

                // The following code performs the delayed update from the left
                // it is optimized for column oriented matrices, but the increased complexity
                // likely causes slower code
                // for (idx_t j = i_pos_block; j < istop_m; ++j)
                // {
                //     idx_t i_bulge_start = (i_pos_last + 2 > j) ? (i_pos_last + 2 - j) / 2 : 0;
                //     for (idx_t i_bulge = i_bulge_start; i_bulge < n_bulges; ++i_bulge)
                //     {
                //         idx_t i_pos = i_pos_last - 2 * i_bulge;
                //         auto v = col(V, i_bulge);
                //         auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                //         A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                //         A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                //         A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                //     }
                // }

                // Delayed update from the left
                for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    for (idx_t j = i_pos + 1; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                    }
                }

                // Accumulate the reflectors into U
                for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    idx_t i1 = (i_pos - i_pos_block) - (i_pos_last - i_pos_block - n_shifts + 2);
                    idx_t i2 = std::min(nrows(U2), (i_pos_last - i_pos_block) + (i_pos_last - i_pos_block - n_shifts + 2) + 3);
                    for (idx_t j = i1; j < i2; ++j)
                    {
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
                idx_t i = i_pos_block + n_block;
                while (i < istop_m)
                {
                    idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                    auto A_slice = slice(A, pair{i_pos_block, i_pos_block + n_block}, pair{i, i + iblock});
                    auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                    lacpy(Uplo::General, WH_slice, A_slice);
                    i = i + iblock;
                }
            }
            // Vertical multiply
            if (istart_m < i_pos_block)
            {
                idx_t i = istart_m;
                while (i < i_pos_block)
                {
                    idx_t iblock = std::min<idx_t>(i_pos_block - i, nrows(WV));
                    auto A_slice = slice(A, pair{i, i + iblock}, pair{i_pos_block, i_pos_block + n_block});
                    auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
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
                    auto Z_slice = slice(Z, pair{i, i + iblock}, pair{i_pos_block, i_pos_block + n_block});
                    auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                    lacpy(Uplo::General, WV_slice, Z_slice);
                    i = i + iblock;
                }
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

            for (idx_t i_pos_last = i_pos_block + n_shifts - 2; i_pos_last < ihi + n_shifts - 1; ++i_pos_last)
            {
                idx_t i_bulge_start = (i_pos_last + 3 > ihi) ? (i_pos_last + 3 - ihi) / 2 : 0;
                for (idx_t i_bulge = i_bulge_start; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    if (i_pos == ihi - 2)
                    {
                        // Special case, the bulge is at the bottom, needs a smaller reflector (order 2)
                        auto v = slice(V, pair{0, 2}, i_bulge);
                        auto h = slice(A, pair{i_pos, i_pos + 2}, i_pos - 1);
                        larfg(h, v[0]);
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
                    }
                    else
                    {
                        auto v = col(V, i_bulge);
                        auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                        move_bulge(H, v, s[size(s)- 1 - 2 * i_bulge], s[size(s)- 1 - 2 * i_bulge - 1]);

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

                        // Apply the reflector we just calculated from the left
                        // We only update a single column, the rest is updated later
                        auto sum = A(i_pos, i_pos) + conj(v[1]) * A(i_pos + 1, i_pos) + conj(v[2]) * A(i_pos + 2, i_pos);
                        A(i_pos, i_pos) = A(i_pos, i_pos) - sum * conj(v[0]);
                        A(i_pos + 1, i_pos) = A(i_pos + 1, i_pos) - sum * conj(v[0]) * v[1];
                        A(i_pos + 2, i_pos) = A(i_pos + 2, i_pos) - sum * conj(v[0]) * v[2];

                        // Test for deflation.
                        if (i_pos > ilo)
                        {
                            if (A(i_pos, i_pos - 1) != zero)
                            {
                                auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                                if (tst1 == zero)
                                {
                                    if (i_pos > ilo + 1)
                                        tst1 += abs1(A(i_pos - 1, i_pos - 2));
                                    if (i_pos > ilo + 2)
                                        tst1 += abs1(A(i_pos - 1, i_pos - 3));
                                    if (i_pos > ilo + 3)
                                        tst1 += abs1(A(i_pos - 1, i_pos - 4));
                                    if (i_pos < ihi - 1)
                                        tst1 += abs1(A(i_pos + 1, i_pos));
                                    if (i_pos < ihi - 2)
                                        tst1 += abs1(A(i_pos + 2, i_pos));
                                    if (i_pos < ihi - 3)
                                        tst1 += abs1(A(i_pos + 3, i_pos));
                                }
                                if (abs1(A(i_pos, i_pos - 1)) < std::max(small_num, eps * tst1))
                                {
                                    auto ab = std::max(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                    auto ba = std::min(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                    auto aa = std::max(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                    auto bb = std::min(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                    auto s = aa + ab;
                                    if (ba * (ab / s) <= std::max(small_num, eps * (bb * (aa / s))))
                                    {
                                        A(i_pos, i_pos - 1) = zero;
                                    }
                                }
                            }
                        }
                    }
                }

                i_bulge_start = (i_pos_last + 4 > ihi) ? (i_pos_last + 4 - ihi) / 2 : 0;

                // The following code performs the delayed update from the left
                // it is optimized for column oriented matrices, but the increased complexity
                // likely causes slower code
                // for (idx_t j = i_pos_block; j < istop_m; ++j)
                // {
                //     idx_t i_bulge_start2 = (i_pos_last + 2 > j) ? (i_pos_last + 2 - j) / 2 : 0;
                //     i_bulge_start2 = std::max(i_bulge_start,i_bulge_start2);
                //     for (idx_t i_bulge = i_bulge_start2; i_bulge < n_bulges; ++i_bulge)
                //     {
                //         idx_t i_pos = i_pos_last - 2 * i_bulge;
                //         auto v = col(V, i_bulge);
                //         auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                //         A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                //         A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                //         A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                //     }
                // }

                // Delayed update from the left
                for (idx_t i_bulge = i_bulge_start; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    for (idx_t j = i_pos + 1; j < istop_m; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                        A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                        A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                    }
                }

                // Accumulate the reflectors into U
                for (idx_t i_bulge = i_bulge_start; i_bulge < n_bulges; ++i_bulge)
                {
                    idx_t i_pos = i_pos_last - 2 * i_bulge;
                    auto v = col(V, i_bulge);
                    idx_t i1 = (i_pos - i_pos_block) - (i_pos_last - i_pos_block - n_shifts + 2);
                    idx_t i2 = std::min(nrows(U2), (i_pos_last - i_pos_block) + (i_pos_last - i_pos_block - n_shifts + 2) + 3);
                    for (idx_t j = i1; j < i2; ++j)
                    {
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
            if (ihi < istop_m)
            {
                idx_t i = ihi;
                while (i < istop_m)
                {
                    idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                    auto A_slice = slice(A, pair{i_pos_block, ihi}, pair{i, i + iblock});
                    auto WH_slice = slice(WH, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::ConjTrans, Op::NoTrans, one, U2, A_slice, zero, WH_slice);
                    lacpy(Uplo::General, WH_slice, A_slice);
                    i = i + iblock;
                }
            }
            // Vertical multiply
            if (istart_m < i_pos_block)
            {
                idx_t i = istart_m;
                while (i < i_pos_block)
                {
                    idx_t iblock = std::min<idx_t>(i_pos_block - i, nrows(WV));
                    auto A_slice = slice(A, pair{i, i + iblock}, pair{i_pos_block, ihi});
                    auto WV_slice = slice(WV, pair{0, nrows(A_slice)}, pair{0, ncols(A_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, A_slice, U2, zero, WV_slice);
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
                    auto Z_slice = slice(Z, pair{i, i + iblock}, pair{i_pos_block, ihi});
                    auto WV_slice = slice(WV, pair{0, nrows(Z_slice)}, pair{0, ncols(Z_slice)});
                    gemm(Op::NoTrans, Op::NoTrans, one, Z_slice, U2, zero, WV_slice);
                    lacpy(Uplo::General, WV_slice, Z_slice);
                    i = i + iblock;
                }
            }
        }
    }

} // lapack

#endif // TLAPACK_QR_SWEEP_HH
