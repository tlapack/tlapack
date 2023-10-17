/// @file multishift_qz_sweep.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_QZ_SWEEP_HH
#define TLAPACK_QZ_SWEEP_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/lahqr_shiftcolumn.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/move_bulge.hpp"

namespace tlapack {

/** multishift_QR_sweep performs a single small-bulge multi-shift QR sweep.
 *
 * @param[in] want_t bool.
 *      If true, the full Schur factor T will be computed.
 *
 * @param[in] want_q bool.
 *      If true, the Schur vectors Q will be computed.
 *
 * @param[in] want_z bool.
 *      If true, the Schur vectors Z will be computed.
 *
 * @param[in] ilo    integer.
 *      Either ilo=0 or A(ilo,ilo-1) = 0.
 *
 * @param[in] ihi    integer.
 *      ilo and ihi determine an isolated block in (A,B).
 *
 * @param[in] A  n by n matrix.
 *      Hessenberg matrix on which AED will be performed
 *
 * @param[in] B  n by n matrix.
 *      Hessenberg matrix on which AED will be performed
 *
 * @param[in] alpha  complex vector.
 *      Vector containing the shifts to be used during the sweep
 *
 * @param[in] beta  vector.
 *      Vector containing the scale factor of the shifts to be used during the
 *      sweep
 *
 * @param[in] Q  n by n matrix.
 *      On entry, the previously calculated Schur factors.
 *
 * @param[in] Z  n by n matrix.
 *      On entry, the previously calculated Schur factors.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR alpha_t,
          TLAPACK_VECTOR beta_t,
          enable_if_t<is_complex<type_t<alpha_t>>, bool> = true>
void multishift_QZ_sweep(bool want_t,
                         bool want_q,
                         bool want_z,
                         size_type<matrix_t> ilo,
                         size_type<matrix_t> ihi,
                         matrix_t& A,
                         matrix_t& B,
                         const alpha_t& alpha,
                         const beta_t& beta,
                         matrix_t& Q,
                         matrix_t& Z)
{
    using TA = type_t<matrix_t>;
    using real_t = real_type<TA>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    const real_t one(1);
    const real_t zero(0);
    const idx_t n = ncols(A);
    const real_t eps = ulp<real_t>();
    const real_t small_num = safe_min<real_t>() * ((real_t)n / eps);

    // Functor
    Create<matrix_t> new_matrix;

    // Define workspace matrices
    // We use the lower triangular part of A and B as workspace
    const idx_t n_block_max = (n - 3) / 3;
    const idx_t n_shifts_max =
        min(ihi - ilo - 1, std::max<idx_t>(2, 3 * (n_block_max / 4)));
    idx_t n_shifts = std::min<idx_t>(size(alpha), n_shifts_max);
    if (n_shifts % 2 == 1) n_shifts = n_shifts - 1;
    idx_t n_bulges = n_shifts / 2;

    const idx_t n_block_desired = std::min<idx_t>(2 * n_shifts, n_block_max);

    // Qc stores the left orthogonal transformations
    auto Qc =
        slice(A, range{n - n_block_desired, n}, range{0, n_block_desired});
    // Zc stores the left orthogonal transformations
    auto Zc =
        slice(B, range{n - n_block_desired, n}, range{0, n_block_desired});

    // Workspace for the reflector (note, this overlaps with the multiplication
    // workspace, they are not needed at the same time)
    auto v = slice(A, range(n - 3, n), n_block_desired);

    // Workspace for horizontal multiplications
    auto WH = slice(A, range{n - n_block_desired, n},
                    range{n_block_desired, n - n_block_desired - 3});

    // Workspace for vertical multiplications
    auto WV = slice(A, range{n_block_desired + 3, n - n_block_desired},
                    range{0, n_block_desired});

    // Position of the shift train in the pencil.
    idx_t i_pos_block;

    //
    // The following code block introduces the bulges into the matrix
    //
    {
        // Near-the-diagonal bulge introduction
        // The calculations are initially limited to a small window.
        // The rest is updated later via level 3 BLAS
        idx_t n_block = min(n_block_desired, ihi - ilo);
        auto A2 =
            slice(A, range(ilo, ilo + n_block), range(ilo, ilo + n_block));
        auto B2 =
            slice(B, range(ilo, ilo + n_block), range(ilo, ilo + n_block));
        auto Qc2 = slice(Qc, range{0, n_block}, range{0, n_block});
        laset(GENERAL, zero, one, Qc2);
        auto Zc2 = slice(Zc, range{0, n_block}, range{0, n_block});
        laset(GENERAL, zero, one, Zc2);

        for (idx_t i_bulge = 0; i_bulge < n_bulges; i_bulge++) {
            TA t1;

            {
                auto H = slice(A2, range{0, 3}, range{0, 3});
                auto T = slice(B2, range{0, 3}, range{0, 3});
                lahqz_shiftcolumn(H, T, v, alpha[2 * i_bulge],
                                  alpha[2 * i_bulge + 1], beta[2 * i_bulge],
                                  beta[2 * i_bulge + 1]);
                larfg(FORWARD, COLUMNWISE_STORAGE, v, t1);
            }

            // Apply reflector to the block
            {
                t1 = conj(t1);
                const TA v2 = v[1];
                const TA t2 = t1 * v2;
                const TA v3 = v[2];
                const TA t3 = t1 * v[2];
                TA sum;

                // Apply reflector from the left to A
                for (idx_t j = 0; j < n_block; ++j) {
                    sum = A2(0, j) + conj(v2) * A2(1, j) + conj(v3) * A2(2, j);
                    A2(0, j) = A2(0, j) - sum * t1;
                    A2(1, j) = A2(1, j) - sum * t2;
                    A2(2, j) = A2(2, j) - sum * t3;
                }
                // Apply reflector from the left to B
                for (idx_t j = 0; j < n_block; ++j) {
                    sum = B2(0, j) + conj(v2) * B2(1, j) + conj(v3) * B2(2, j);
                    B2(0, j) = B2(0, j) - sum * t1;
                    B2(1, j) = B2(1, j) - sum * t2;
                    B2(2, j) = B2(2, j) - sum * t3;
                }
                // Apply reflector to Qc from the right
                for (idx_t j = 0; j < n_block; ++j) {
                    sum = Qc2(j, 0) + v2 * Qc2(j, 1) + v3 * Qc2(j, 2);
                    Qc2(j, 0) = Qc2(j, 0) - sum * conj(t1);
                    Qc2(j, 1) = Qc2(j, 1) - sum * conj(t2);
                    Qc2(j, 2) = Qc2(j, 2) - sum * conj(t3);
                }
            }

            // Move the bulge down to make room for another bulge
            for (idx_t i = 0; i < n_block - 3 - 2 * i_bulge; ++i) {
                // Remove fill-in from B using an inverse reflector
                {
                    auto T = slice(B2, range{i, i + 3}, range{i, i + 3});
                    inv_house3(T, v, t1);
                }

                // Apply the reflector
                {
                    t1 = conj(t1);
                    const TA v2 = v[1];
                    const TA t2 = t1 * v2;
                    const TA v3 = v[2];
                    const TA t3 = t1 * v[2];
                    TA sum;

                    // Apply reflector from the right to B
                    for (idx_t j = 0; j < i + 3; ++j) {
                        sum = B2(j, i) + v2 * B2(j, i + 1) + v3 * B2(j, i + 2);
                        B2(j, i) = B2(j, i) - sum * conj(t1);
                        B2(j, i + 1) = B2(j, i + 1) - sum * conj(t2);
                        B2(j, i + 2) = B2(j, i + 2) - sum * conj(t3);
                    }
                    B2(i + 1, i) = (TA)0;
                    B2(i + 2, i) = (TA)0;
                    // Apply reflector from the right to A
                    for (idx_t j = 0; j < min(i + 4, n_block); ++j) {
                        sum = A2(j, i) + v2 * A2(j, i + 1) + v3 * A2(j, i + 2);
                        A2(j, i) = A2(j, i) - sum * conj(t1);
                        A2(j, i + 1) = A2(j, i + 1) - sum * conj(t2);
                        A2(j, i + 2) = A2(j, i + 2) - sum * conj(t3);
                    }
                    // Apply reflector to Zc from the right
                    for (idx_t j = 0; j < n_block; ++j) {
                        sum =
                            Zc2(j, i) + v2 * Zc2(j, i + 1) + v3 * Zc2(j, i + 2);
                        Zc2(j, i) = Zc2(j, i) - sum * conj(t1);
                        Zc2(j, i + 1) = Zc2(j, i + 1) - sum * conj(t2);
                        Zc2(j, i + 2) = Zc2(j, i + 2) - sum * conj(t3);
                    }
                }

                // Calculate a reflector to move the bulge one position
                v[0] = A2(i + 1, i);
                v[1] = A2(i + 2, i);
                v[2] = A2(i + 3, i);
                larfg(FORWARD, COLUMNWISE_STORAGE, v, t1);
                A2(i + 1, i) = v[0];
                A2(i + 2, i) = zero;
                A2(i + 3, i) = zero;

                // Apply reflector
                {
                    t1 = conj(t1);
                    const TA v2 = v[1];
                    const TA t2 = t1 * v2;
                    const TA v3 = v[2];
                    const TA t3 = t1 * v[2];
                    TA sum;

                    // Apply reflector from the left to A
                    for (idx_t j = i + 1; j < n_block; ++j) {
                        sum = A2(i + 1, j) + conj(v2) * A2(i + 2, j) +
                              conj(v3) * A2(i + 3, j);
                        A2(i + 1, j) = A2(i + 1, j) - sum * t1;
                        A2(i + 2, j) = A2(i + 2, j) - sum * t2;
                        A2(i + 3, j) = A2(i + 3, j) - sum * t3;
                    }
                    // Apply reflector from the left to B
                    for (idx_t j = i + 1; j < n_block; ++j) {
                        sum = B2(i + 1, j) + conj(v2) * B2(i + 2, j) +
                              conj(v3) * B2(i + 3, j);
                        B2(i + 1, j) = B2(i + 1, j) - sum * t1;
                        B2(i + 2, j) = B2(i + 2, j) - sum * t2;
                        B2(i + 3, j) = B2(i + 3, j) - sum * t3;
                    }
                    // Apply reflector to Qc from the right
                    for (idx_t j = 0; j < n_block; ++j) {
                        sum = Qc2(j, i + 1) + v2 * Qc2(j, i + 2) +
                              v3 * Qc2(j, i + 3);
                        Qc2(j, i + 1) = Qc2(j, i + 1) - sum * conj(t1);
                        Qc2(j, i + 2) = Qc2(j, i + 2) - sum * conj(t2);
                        Qc2(j, i + 3) = Qc2(j, i + 3) - sum * conj(t3);
                    }
                }
            }
        }
        // Update rest of the matrix
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
        if (ilo + n_shifts + 1 < istop_m) {
            idx_t i = ilo + n_block;
            while (i < istop_m) {
                idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                auto A_slice =
                    slice(A, range{ilo, ilo + n_block}, range{i, i + iblock});
                auto WH_slice = slice(WH, range{0, nrows(A_slice)},
                                      range{0, ncols(A_slice)});
                gemm(CONJ_TRANS, NO_TRANS, one, Qc2, A_slice, WH_slice);
                lacpy(GENERAL, WH_slice, A_slice);
                i = i + iblock;
            }
        }
        if (istart_m < ilo) {
            idx_t i = istart_m;
            while (i < ilo) {
                idx_t iblock = std::min<idx_t>(ilo - i, nrows(WV));
                auto A_slice =
                    slice(A, range{i, i + iblock}, range{ilo, ilo + n_block});
                auto WV_slice = slice(WV, range{0, nrows(A_slice)},
                                      range{0, ncols(A_slice)});
                gemm(NO_TRANS, NO_TRANS, one, A_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, A_slice);
                i = i + iblock;
            }
        }
        // Update B
        if (ilo + n_shifts + 1 < istop_m) {
            idx_t i = ilo + n_block;
            while (i < istop_m) {
                idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                auto B_slice =
                    slice(B, range{ilo, ilo + n_block}, range{i, i + iblock});
                auto WH_slice = slice(WH, range{0, nrows(B_slice)},
                                      range{0, ncols(B_slice)});
                gemm(CONJ_TRANS, NO_TRANS, one, Qc2, B_slice, WH_slice);
                lacpy(GENERAL, WH_slice, B_slice);
                i = i + iblock;
            }
        }
        if (istart_m < ilo) {
            idx_t i = istart_m;
            while (i < ilo) {
                idx_t iblock = std::min<idx_t>(ilo - i, nrows(WV));
                auto B_slice =
                    slice(B, range{i, i + iblock}, range{ilo, ilo + n_block});
                auto WV_slice = slice(WV, range{0, nrows(B_slice)},
                                      range{0, ncols(B_slice)});
                gemm(NO_TRANS, NO_TRANS, one, B_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, B_slice);
                i = i + iblock;
            }
        }
        // Update Q
        if (want_q) {
            idx_t i = 0;
            while (i < n) {
                idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
                auto Q_slice =
                    slice(Q, range{i, i + iblock}, range{ilo, ilo + n_block});
                auto WV_slice = slice(WV, range{0, nrows(Q_slice)},
                                      range{0, ncols(Q_slice)});
                gemm(NO_TRANS, NO_TRANS, one, Q_slice, Qc2, WV_slice);
                lacpy(GENERAL, WV_slice, Q_slice);
                i = i + iblock;
            }
        }
        // Update Z
        if (want_z) {
            idx_t i = 0;
            while (i < n) {
                idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
                auto Z_slice =
                    slice(Z, range{i, i + iblock}, range{ilo, ilo + n_block});
                auto WV_slice = slice(WV, range{0, nrows(Z_slice)},
                                      range{0, ncols(Z_slice)});
                gemm(NO_TRANS, NO_TRANS, one, Z_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, Z_slice);
                i = i + iblock;
            }
        }

        i_pos_block = ilo + n_block - n_shifts - 1;
    }

    //
    // The following code block moves the bulges down until they are low enough
    // to be removed
    //
    while (i_pos_block + n_block_desired + 1 < ihi) {
        // Number of positions each bulge will be moved down
        idx_t n_pos = std::min<idx_t>(n_block_desired - n_shifts,
                                      ihi - n_shifts - 1 - i_pos_block);
        // Actual blocksize
        idx_t n_block = n_shifts + n_pos;

        auto Qc2 = slice(Qc, range{0, n_block}, range{0, n_block});
        laset(GENERAL, zero, one, Qc2);
        auto Zc2 = slice(Zc, range{0, n_block}, range{0, n_block});
        laset(GENERAL, zero, one, Zc2);

        // Near-the-diagonal bulge chase
        // The calculations are initially limited to a small window
        // The rest is updated later via level 3 BLAS
        auto A2 = slice(A, range(i_pos_block, i_pos_block + 1 + n_block),
                        range(i_pos_block, i_pos_block + n_block));
        auto B2 = slice(B, range(i_pos_block, i_pos_block + 1 + n_block),
                        range(i_pos_block, i_pos_block + n_block));

        for (idx_t i_bulge = 0; i_bulge < n_bulges; i_bulge++) {
            TA t1;
            idx_t i2 = 2 * (n_bulges - i_bulge - 1);
            for (idx_t i = i2; i < i2 + n_pos; ++i) {
                // Remove fill-in from B using an inverse reflector
                {
                    auto T = slice(B2, range{i, i + 3}, range{i, i + 3});
                    inv_house3(T, v, t1);
                }

                // Apply the reflector
                {
                    t1 = conj(t1);
                    const TA v2 = v[1];
                    const TA t2 = t1 * v2;
                    const TA v3 = v[2];
                    const TA t3 = t1 * v[2];
                    TA sum;

                    // Apply reflector from the right to B
                    for (idx_t j = 0; j < i + 3; ++j) {
                        sum = B2(j, i) + v2 * B2(j, i + 1) + v3 * B2(j, i + 2);
                        B2(j, i) = B2(j, i) - sum * conj(t1);
                        B2(j, i + 1) = B2(j, i + 1) - sum * conj(t2);
                        B2(j, i + 2) = B2(j, i + 2) - sum * conj(t3);
                    }
                    B2(i + 1, i) = (TA)0;
                    B2(i + 2, i) = (TA)0;
                    // Apply reflector from the right to A
                    for (idx_t j = 0; j < i + 4; ++j) {
                        sum = A2(j, i) + v2 * A2(j, i + 1) + v3 * A2(j, i + 2);
                        A2(j, i) = A2(j, i) - sum * conj(t1);
                        A2(j, i + 1) = A2(j, i + 1) - sum * conj(t2);
                        A2(j, i + 2) = A2(j, i + 2) - sum * conj(t3);
                    }
                    // Apply reflector to Zc from the right
                    for (idx_t j = 0; j < n_block; ++j) {
                        sum =
                            Zc2(j, i) + v2 * Zc2(j, i + 1) + v3 * Zc2(j, i + 2);
                        Zc2(j, i) = Zc2(j, i) - sum * conj(t1);
                        Zc2(j, i + 1) = Zc2(j, i + 1) - sum * conj(t2);
                        Zc2(j, i + 2) = Zc2(j, i + 2) - sum * conj(t3);
                    }
                }

                // Calculate a reflector to move the bulge one position
                v[0] = A2(i + 1, i);
                v[1] = A2(i + 2, i);
                v[2] = A2(i + 3, i);
                larfg(FORWARD, COLUMNWISE_STORAGE, v, t1);
                A2(i + 1, i) = v[0];
                A2(i + 2, i) = zero;
                A2(i + 3, i) = zero;

                // Apply reflector
                {
                    t1 = conj(t1);
                    const TA v2 = v[1];
                    const TA t2 = t1 * v2;
                    const TA v3 = v[2];
                    const TA t3 = t1 * v[2];
                    TA sum;

                    // Apply reflector from the left to A
                    for (idx_t j = i + 1; j < n_block; ++j) {
                        sum = A2(i + 1, j) + conj(v2) * A2(i + 2, j) +
                              conj(v3) * A2(i + 3, j);
                        A2(i + 1, j) = A2(i + 1, j) - sum * t1;
                        A2(i + 2, j) = A2(i + 2, j) - sum * t2;
                        A2(i + 3, j) = A2(i + 3, j) - sum * t3;
                    }
                    // Apply reflector from the left to B
                    for (idx_t j = i + 1; j < n_block; ++j) {
                        sum = B2(i + 1, j) + conj(v2) * B2(i + 2, j) +
                              conj(v3) * B2(i + 3, j);
                        B2(i + 1, j) = B2(i + 1, j) - sum * t1;
                        B2(i + 2, j) = B2(i + 2, j) - sum * t2;
                        B2(i + 3, j) = B2(i + 3, j) - sum * t3;
                    }
                    // Apply reflector to Qc from the right
                    for (idx_t j = 0; j < n_block; ++j) {
                        sum =
                            Qc2(j, i) + v2 * Qc2(j, i + 1) + v3 * Qc2(j, i + 2);
                        Qc2(j, i) = Qc2(j, i) - sum * conj(t1);
                        Qc2(j, i + 1) = Qc2(j, i + 1) - sum * conj(t2);
                        Qc2(j, i + 2) = Qc2(j, i + 2) - sum * conj(t3);
                    }
                }
            }
        }
        // Update rest of the matrix
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
        if (i_pos_block + n_block < istop_m) {
            idx_t i = i_pos_block + n_block;
            while (i < istop_m) {
                idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                auto A_slice =
                    slice(A, range{i_pos_block + 1, i_pos_block + 1 + n_block},
                          range{i, i + iblock});
                auto WH_slice = slice(WH, range{0, nrows(A_slice)},
                                      range{0, ncols(A_slice)});
                gemm(CONJ_TRANS, NO_TRANS, one, Qc2, A_slice, WH_slice);
                lacpy(GENERAL, WH_slice, A_slice);
                i = i + iblock;
            }
        }
        if (istart_m < i_pos_block) {
            idx_t i = istart_m;
            while (i < i_pos_block) {
                idx_t iblock = std::min<idx_t>(i_pos_block - i, nrows(WV));
                auto A_slice = slice(A, range{i, i + iblock},
                                     range{i_pos_block, i_pos_block + n_block});
                auto WV_slice = slice(WV, range{0, nrows(A_slice)},
                                      range{0, ncols(A_slice)});
                gemm(NO_TRANS, NO_TRANS, one, A_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, A_slice);
                i = i + iblock;
            }
        }
        // Update B
        if (i_pos_block + n_block < istop_m) {
            idx_t i = i_pos_block + n_block;
            while (i < istop_m) {
                idx_t iblock = std::min<idx_t>(istop_m - i, ncols(WH));
                auto B_slice =
                    slice(B, range{i_pos_block + 1, i_pos_block + 1 + n_block},
                          range{i, i + iblock});
                auto WH_slice = slice(WH, range{0, nrows(B_slice)},
                                      range{0, ncols(B_slice)});
                gemm(CONJ_TRANS, NO_TRANS, one, Qc2, B_slice, WH_slice);
                lacpy(GENERAL, WH_slice, B_slice);
                i = i + iblock;
            }
        }
        if (istart_m < i_pos_block) {
            idx_t i = istart_m;
            while (i < i_pos_block) {
                idx_t iblock = std::min<idx_t>(i_pos_block - i, nrows(WV));
                auto B_slice = slice(B, range{i, i + iblock},
                                     range{i_pos_block, i_pos_block + n_block});
                auto WV_slice = slice(WV, range{0, nrows(B_slice)},
                                      range{0, ncols(B_slice)});
                gemm(NO_TRANS, NO_TRANS, one, B_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, B_slice);
                i = i + iblock;
            }
        }
        // Update Q
        if (want_q) {
            idx_t i = 0;
            while (i < n) {
                idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
                auto Q_slice =
                    slice(Q, range{i, i + iblock},
                          range{i_pos_block + 1, i_pos_block + 1 + n_block});
                auto WV_slice = slice(WV, range{0, nrows(Q_slice)},
                                      range{0, ncols(Q_slice)});
                gemm(NO_TRANS, NO_TRANS, one, Q_slice, Qc2, WV_slice);
                lacpy(GENERAL, WV_slice, Q_slice);
                i = i + iblock;
            }
        }
        // Update Z
        if (want_z) {
            idx_t i = 0;
            while (i < n) {
                idx_t iblock = std::min<idx_t>(n - i, nrows(WV));
                auto Z_slice = slice(Z, range{i, i + iblock},
                                     range{i_pos_block, i_pos_block + n_block});
                auto WV_slice = slice(WV, range{0, nrows(Z_slice)},
                                      range{0, ncols(Z_slice)});
                gemm(NO_TRANS, NO_TRANS, one, Z_slice, Zc2, WV_slice);
                lacpy(GENERAL, WV_slice, Z_slice);
                i = i + iblock;
            }
        }

        i_pos_block = i_pos_block + n_pos;
    }
}

}  // namespace tlapack

#endif  // TLAPACK_QZ_SWEEP_HH
