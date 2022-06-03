/// @file recursive_bulge_move.hpp
/// @author Thijs Steel, KU Leuven, Belgium
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __RECURSIVE_BULGE_MOVE_HH__
#define __RECURSIVE_BULGE_MOVE_HH__

#include <memory>
#include <complex>

#include "legacy_api/base/utils.hpp"
#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/larfg.hpp"
#include "lapack/lahqr_shiftcolumn.hpp"
#include "lapack/move_bulge.hpp"

namespace tlapack
{

    template <typename idx_t, typename T>
    struct move_bulges_opts_t
    {
        // Optimization parameter, recursion will be ended if
        // number of shifts or number of positions is smaller than nx.
        idx_t nx = 32;
        // Workspace pointer, if no workspace is provided, one will be allocated internally
        T *_work = nullptr;
        // Workspace size
        idx_t lwork;
    };

    /** move_bulges pushes shifts present in the pencil down to the edge of the matrix.
     *
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix with bulges present in the first size(s) positions.
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[out] Q  n-1 by n-1 matrix.
     *      The orthogonal matrix Q
     *
     * @param[in,out] V    3 by size(s)/2 matrix.
     *      Matrix containing delayed reflectors.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void move_bulges_recursive(matrix_t &A, vector_t &s, matrix_t &Q, matrix_t &V,
                               const move_bulges_opts_t<size_type<matrix_t>, type_t<matrix_t>> &opts = {})
    {

        using T = type_t<matrix_t>;
        using real_t = real_type<T>;
        using idx_t = size_type<matrix_t>;
        using pair = std::pair<idx_t, idx_t>;

        const idx_t n = ncols(A);

        const idx_t ns = size(s);
        const idx_t np = n - 1 - ns;

        if (min(ns, np) <= opts.nx)
        {
            return move_bulges(A, s, Q, V);
        }

        const idx_t nb = ns / 2;

        // Split the bulges into two group, move each of them
        // np1 positions and then np2 positions
        const idx_t nb1 = nb / 2;
        const idx_t nb2 = nb - nb1;
        const idx_t np1 = np / 2;
        const idx_t np2 = np - np1;

        // Get workspace
        T *_work;
        idx_t lwork;
        idx_t required_workspace = (n + 2 * nb2 + np2) * (2 * nb2 + np2);
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
            _work = new T[lwork];
        }
        idx_t iwork2 = (2 * nb2 + np2) * (2 * nb2 + np2);
        idx_t lwork2 = lwork - iwork2;

        move_bulges_opts_t<idx_t, T> opts2;
        opts2.nx = opts.nx;
        opts2._work = &_work[iwork2];
        opts2.lwork = lwork2;

        laset(Uplo::General, (T)0, (T)1, Q);

        // Move last group of bulges np1 positions
        {
            idx_t i_pos = 2 * nb1;
            idx_t n_block = 2 * nb2 + np1 + 1;
            auto Q_work = legacyMatrix<T, layout<matrix_t>>(n_block - 1, n_block - 1, &_work[0], n_block - 1);
            auto A_slice = slice(A, pair{i_pos, i_pos + n_block}, pair{i_pos, i_pos + n_block});
            auto V_slice = slice(V, pair{0, 3}, pair{0, nb2});
            auto s_slice = slice(s, pair{0, 2 * nb2});
            move_bulges_recursive(A_slice, s_slice, Q_work, V_slice, opts2);

            // Update Q
            auto Q_slice = slice(Q, pair{i_pos, i_pos + n_block - 1}, pair{i_pos, i_pos + n_block - 1});
            lacpy(Uplo::General, Q_work, Q_slice);

            // Multiply A with Q_work from the left
            auto A_slice_left = slice(A, pair{i_pos + 1, i_pos + n_block}, pair{i_pos + n_block, n});
            auto work_left = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_left), ncols(A_slice_left),
                                                               &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_left) : ncols(A_slice_left));
            gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q_work, A_slice_left, (real_t)0.0, work_left);
            lacpy(Uplo::General, work_left, A_slice_left);

            // Multiply A with Q_work from the right
            auto A_slice_right = slice(A, pair{0, i_pos}, pair{i_pos + 1, i_pos + n_block});
            auto work_right = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_right), ncols(A_slice_right),
                                                                &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_right) : ncols(A_slice_right));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, A_slice_right, Q_work, (real_t)0.0, work_right);
            lacpy(Uplo::General, work_right, A_slice_right);
        }

        // Move last group of bulges np2 positions
        {
            idx_t i_pos = 2 * nb1 + np1;
            idx_t n_block = 2 * nb2 + np2 + 1;
            auto Q_work = legacyMatrix<T, layout<matrix_t>>(n_block - 1, n_block - 1, &_work[0], n_block - 1);
            auto A_slice = slice(A, pair{i_pos, i_pos + n_block}, pair{i_pos, i_pos + n_block});
            auto V_slice = slice(V, pair{0, 3}, pair{0, nb2});
            auto s_slice = slice(s, pair{0, 2 * nb2});
            move_bulges_recursive(A_slice, s_slice, Q_work, V_slice, opts2);

            // Update Q
            auto Q_slice = slice(Q, pair{0, nrows(Q)}, pair{i_pos, i_pos + n_block - 1});
            auto Q_work_right = legacyMatrix<T, layout<matrix_t>>(nrows(Q_slice), ncols(Q_slice),
                                                                  &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(Q_slice) : ncols(Q_slice));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q_slice, Q_work, (real_t)0.0, Q_work_right);
            lacpy(Uplo::General, Q_work_right, Q_slice);

            // Multiply A with Q_work from the right
            auto A_slice_right = slice(A, pair{0, i_pos}, pair{i_pos + 1, i_pos + n_block});
            auto work_right = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_right), ncols(A_slice_right),
                                                                &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_right) : ncols(A_slice_right));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, A_slice_right, Q_work, (real_t)0.0, work_right);
            lacpy(Uplo::General, work_right, A_slice_right);
        }

        // Move first group of bulges np1 positions
        {
            idx_t i_pos = 0;
            idx_t n_block = 2 * nb1 + np1 + 1;
            auto Q_work = legacyMatrix<T, layout<matrix_t>>(n_block - 1, n_block - 1, &_work[0], n_block - 1);
            auto A_slice = slice(A, pair{i_pos, i_pos + n_block}, pair{i_pos, i_pos + n_block});
            auto V_slice = slice(V, pair{0, 3}, pair{nb2, nb});
            auto s_slice = slice(s, pair{2 * nb2, 2 * nb});
            move_bulges_recursive(A_slice, s_slice, Q_work, V_slice, opts2);

            // Update Q
            auto Q_slice = slice(Q, pair{0, nrows(Q)}, pair{i_pos, i_pos + n_block - 1});
            auto Q_work_right = legacyMatrix<T, layout<matrix_t>>(nrows(Q_slice), ncols(Q_slice),
                                                                  &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(Q_slice) : ncols(Q_slice));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q_slice, Q_work, (real_t)0.0, Q_work_right);
            lacpy(Uplo::General, Q_work_right, Q_slice);

            // Multiply A with Q_work from the left
            auto A_slice_left = slice(A, pair{i_pos + 1, i_pos + n_block}, pair{i_pos + n_block, n});
            auto work_left = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_left), ncols(A_slice_left),
                                                               &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_left) : ncols(A_slice_left));
            gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q_work, A_slice_left, (real_t)0.0, work_left);
            lacpy(Uplo::General, work_left, A_slice_left);
        }

        // Move last group of bulges np2 positions
        {
            idx_t i_pos = np1;
            idx_t n_block = 2 * nb1 + np2 + 1;
            auto Q_work = legacyMatrix<T, layout<matrix_t>>(n_block - 1, n_block - 1, &_work[0], n_block - 1);
            auto A_slice = slice(A, pair{i_pos, i_pos + n_block}, pair{i_pos, i_pos + n_block});
            auto V_slice = slice(V, pair{0, 3}, pair{nb2, nb});
            auto s_slice = slice(s, pair{2 * nb2, 2 * nb});
            move_bulges_recursive(A_slice, s_slice, Q_work, V_slice, opts2);

            // Update Q
            auto Q_slice = slice(Q, pair{0, nrows(Q)}, pair{i_pos, i_pos + n_block - 1});
            auto Q_work_right = legacyMatrix<T, layout<matrix_t>>(nrows(Q_slice), ncols(Q_slice),
                                                                  &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(Q_slice) : ncols(Q_slice));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, Q_slice, Q_work, (real_t)0.0, Q_work_right);
            lacpy(Uplo::General, Q_work_right, Q_slice);

            // Multiply A with Q_work from the left
            auto A_slice_left = slice(A, pair{i_pos + 1, i_pos + n_block}, pair{i_pos + n_block, n});
            auto work_left = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_left), ncols(A_slice_left),
                                                               &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_left) : ncols(A_slice_left));
            gemm(Op::ConjTrans, Op::NoTrans, (real_t)1.0, Q_work, A_slice_left, (real_t)0.0, work_left);
            lacpy(Uplo::General, work_left, A_slice_left);

            // Multiply A with Q_work from the right
            auto A_slice_right = slice(A, pair{0, i_pos}, pair{i_pos + 1, i_pos + n_block});
            auto work_right = legacyMatrix<T, layout<matrix_t>>(nrows(A_slice_right), ncols(A_slice_right),
                                                                &_work[iwork2], layout<matrix_t> == Layout::ColMajor ? nrows(A_slice_right) : ncols(A_slice_right));
            gemm(Op::NoTrans, Op::NoTrans, (real_t)1.0, A_slice_right, Q_work, (real_t)0.0, work_right);
            lacpy(Uplo::General, work_right, A_slice_right);
        }

        if (locally_allocated)
            delete[] _work;
    }

    /** move_bulges pushes shifts present in the pencil down to the edge of the matrix.
     *
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix with bulges present in the first size(s) positions.
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[out] Q  n-1 by n-1 matrix.
     *      The orthogonal matrix Q
     *
     * @param[in,out] V    3 by size(s)/2 matrix.
     *      Matrix containing delayed reflectors.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void move_bulges(matrix_t &A, vector_t &s, matrix_t &Q, matrix_t &V)
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
        const idx_t ns = size(s);
        const idx_t n_bulges = ns / 2;

        // Number of positions to move the bulges
        const idx_t np = n - 1 - ns;

        laset(Uplo::General, zero, one, Q);

        for (idx_t ip = 0; ip < np; ++ip)
        {
            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                idx_t i_pos = 1 + ip + 2 * (n_bulges - i_bulge - 1);
                auto v = col(V, i_bulge);
                auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                // Apply the reflector we just calculated from the right
                // We leave the last row for later (it interferes with the optimally packed bulges)
                auto t0 = v[0];
                auto t1 = t0 * conj(v[1]);
                auto t2 = t0 * conj(v[2]);
                for (idx_t j = 0; j < i_pos + 3; ++j)
                {
                    auto sum = A(j, i_pos) + v[1] * A(j, i_pos + 1) + v[2] * A(j, i_pos + 2);
                    A(j, i_pos) = A(j, i_pos) - sum * t0;
                    A(j, i_pos + 1) = A(j, i_pos + 1) - sum * t1;
                    A(j, i_pos + 2) = A(j, i_pos + 2) - sum * t2;
                }

                // Apply the reflector we just calculated from the left
                // We only update a single column, the rest is updated later
                auto sum = A(i_pos, i_pos) + conj(v[1]) * A(i_pos + 1, i_pos) + conj(v[2]) * A(i_pos + 2, i_pos);
                A(i_pos, i_pos) = A(i_pos, i_pos) - sum * conj(v[0]);
                A(i_pos + 1, i_pos) = A(i_pos + 1, i_pos) - sum * conj(v[0]) * v[1];
                A(i_pos + 2, i_pos) = A(i_pos + 2, i_pos) - sum * conj(v[0]) * v[2];

                // Test for deflation.
                if (A(i_pos, i_pos - 1) != zero)
                {
                    auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                    if (tst1 == zero)
                    {
                        if (i_pos > 1)
                            tst1 += abs1(A(i_pos - 1, i_pos - 2));
                        if (i_pos > 2)
                            tst1 += abs1(A(i_pos - 1, i_pos - 3));
                        if (i_pos > 3)
                            tst1 += abs1(A(i_pos - 1, i_pos - 4));
                        if (i_pos < n - 1)
                            tst1 += abs1(A(i_pos + 1, i_pos));
                        if (i_pos < n - 2)
                            tst1 += abs1(A(i_pos + 2, i_pos));
                        if (i_pos < n - 3)
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
                idx_t i_pos = 1 + ip + 2 * (n_bulges - i_bulge - 1);
                auto v = col(V, i_bulge);
                auto t0 = conj(v[0]);
                auto t1 = t0 * v[1];
                auto t2 = t0 * v[2];
                for (idx_t j = i_pos + 1; j < n; ++j)
                {
                    auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                    A(i_pos, j) = A(i_pos, j) - sum * t0;
                    A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t1;
                    A(i_pos + 2, j) = A(i_pos + 2, j) - sum * t2;
                }
            }

            // Accumulate the reflectors into U
            for (idx_t i_bulge = 0; i_bulge < n_bulges; ++i_bulge)
            {
                idx_t i_pos = ip + 2 * (n_bulges - i_bulge - 1);
                auto v = col(V, i_bulge);
                // idx_t i1 = (i_pos - i_pos_block) - (i_pos_last - i_pos_block - n_shifts + 2);
                // idx_t i2 = std::min(nrows(U2), (i_pos_last - i_pos_block) + (i_pos_last - i_pos_block - n_shifts + 2) + 3);
                auto t0 = v[0];
                auto t1 = t0 * conj(v[1]);
                auto t2 = t0 * conj(v[2]);
                for (idx_t j = 0; j < ncols(Q); ++j)
                {
                    auto sum = Q(j, i_pos) + v[1] * Q(j, i_pos + 1) + v[2] * Q(j, i_pos + 2);
                    Q(j, i_pos) = Q(j, i_pos) - sum * t0;
                    Q(j, i_pos + 1) = Q(j, i_pos + 1) - sum * t1;
                    Q(j, i_pos + 2) = Q(j, i_pos + 2) - sum * t2;
                }
            }
        }

        return;
    }

    /** introduce_bulges introduces bulges into the matrix A and moves them down just enough to
     *  make room for the other bulges.
     *
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix with bulges present in the first size(s) positions.
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[out] Q  n-1 by n-1 matrix.
     *      The orthogonal matrix Q
     *
     * @param[out] V    3 by size(s)/2 matrix.
     *      Matrix containing delayed reflectors.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void introduce_bulges(matrix_t &A, vector_t &s, matrix_t &Q, matrix_t &V)
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
        const idx_t ns = size(s);
        const idx_t n_bulges = ns / 2;

        laset(Uplo::General, zero, one, Q);

        for (idx_t i_pos_last = 0; i_pos_last < ns - 1; ++i_pos_last)
        {
            // The number of bulges that are in the pencil
            idx_t n_active_bulges = min(n_bulges, (i_pos_last / 2) + 1);
            for (idx_t i_bulge = 0; i_bulge < n_active_bulges; ++i_bulge)
            {
                idx_t i_pos = i_pos_last - 2 * i_bulge;
                auto v = col(V, i_bulge);
                if (i_pos == 0)
                {
                    // Introduce bulge
                    T tau;
                    auto H = slice(A, pair{0, 3}, pair{0, 3});
                    lahqr_shiftcolumn(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);
                    larfg(v, tau);
                    v[0] = tau;
                }
                else
                {
                    // Chase bulge down
                    auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                    move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);
                }

                // Apply the reflector we just calculated from the right
                // We leave the last row for later (it interferes with the optimally packed bulges)
                for (idx_t j = 0; j < i_pos + 3; ++j)
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
                if (i_pos > 0)
                {
                    if (A(i_pos, i_pos - 1) != zero)
                    {
                        auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                        if (tst1 == zero)
                        {
                            if (i_pos > 1)
                                tst1 += abs1(A(i_pos - 1, i_pos - 2));
                            if (i_pos > 2)
                                tst1 += abs1(A(i_pos - 1, i_pos - 3));
                            if (i_pos > 3)
                                tst1 += abs1(A(i_pos - 1, i_pos - 4));
                            if (i_pos < n - 1)
                                tst1 += abs1(A(i_pos + 1, i_pos));
                            if (i_pos < n - 2)
                                tst1 += abs1(A(i_pos + 2, i_pos));
                            if (i_pos < n - 3)
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
                for (idx_t j = i_pos + 1; j < n; ++j)
                {
                    auto sum = A(i_pos, j) + conj(v[1]) * A(i_pos + 1, j) + conj(v[2]) * A(i_pos + 2, j);
                    A(i_pos, j) = A(i_pos, j) - sum * conj(v[0]);
                    A(i_pos + 1, j) = A(i_pos + 1, j) - sum * conj(v[0]) * v[1];
                    A(i_pos + 2, j) = A(i_pos + 2, j) - sum * conj(v[0]) * v[2];
                }
            }

            // Accumulate the reflectors into Q
            for (idx_t i_bulge = 0; i_bulge < n_active_bulges; ++i_bulge)
            {
                idx_t i_pos = i_pos_last - 2 * i_bulge;
                auto v = col(V, i_bulge);
                idx_t i1 = 0;
                idx_t i2 = min(nrows(Q), 2 * i_pos_last + 3);
                for (idx_t j = i1; j < i2; ++j)
                {
                    auto sum = Q(j, i_pos) + v[1] * Q(j, i_pos + 1) + v[2] * Q(j, i_pos + 2);
                    Q(j, i_pos) = Q(j, i_pos) - sum * v[0];
                    Q(j, i_pos + 1) = Q(j, i_pos + 1) - sum * v[0] * conj(v[1]);
                    Q(j, i_pos + 2) = Q(j, i_pos + 2) - sum * v[0] * conj(v[2]);
                }
            }
        }

        return;
    }

    /** remove bulges removes bulges from the matrix A
     *
     * @param[in,out] A  n by n matrix.
     *      Hessenberg matrix with bulges present in the last size(s) positions.
     *
     * @param[in] s  complex vector.
     *      Vector containing the shifts to be used during the sweep
     *
     * @param[out] Q  n-1 by n-1 matrix.
     *      The orthogonal matrix Q
     *
     * @param[out] V    3 by size(s)/2 matrix.
     *      Matrix containing delayed reflectors.
     *
     * @ingroup geev
     */
    template <
        class matrix_t,
        class vector_t,
        enable_if_t<is_complex<type_t<vector_t>>::value, bool> = true>
    void remove_bulges(matrix_t &A, vector_t &s, matrix_t &Q, matrix_t &V)
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
        const idx_t ns = size(s);
        const idx_t n_bulges = ns / 2;

        laset(Uplo::General, zero, one, Q);

        for (idx_t i_pos_last = n - 2; i_pos_last < n + ns; ++i_pos_last)
        {
            idx_t i_bulge_start = (i_pos_last + 3 > n) ? (i_pos_last + 3 - n) / 2 : 0;
            for (idx_t i_bulge = i_bulge_start; i_bulge < n_bulges; ++i_bulge)
            {
                idx_t i_pos = i_pos_last - 2 * i_bulge;
                if (i_pos == n - 2)
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
                    for (idx_t j = 0; j < i_pos + 2; ++j)
                    {
                        auto sum = A(j, i_pos) + v2 * A(j, i_pos + 1);
                        A(j, i_pos) = A(j, i_pos) - sum * conj(t1);
                        A(j, i_pos + 1) = A(j, i_pos + 1) - sum * conj(t2);
                    }
                    // Apply the reflector we just calculated from the left
                    for (idx_t j = i_pos; j < n; ++j)
                    {
                        auto sum = A(i_pos, j) + conj(v2) * A(i_pos + 1, j);
                        A(i_pos, j) = A(i_pos, j) - sum * t1;
                        A(i_pos + 1, j) = A(i_pos + 1, j) - sum * t2;
                    }
                    // Accumulate the reflector into U
                    // The loop bounds should be changed to reflect the fact that U2 starts off as diagonal
                    for (idx_t j = 0; j < nrows(Q); ++j)
                    {
                        auto sum = Q(j, i_pos - 1) + v2 * Q(j, i_pos);
                        Q(j, i_pos - 1) = Q(j, i_pos - 1) - sum * conj(t1);
                        Q(j, i_pos) = Q(j, i_pos) - sum * conj(t2);
                    }
                }
                else
                {
                    auto v = col(V, i_bulge);
                    auto H = slice(A, pair{i_pos - 1, i_pos + 3}, pair{i_pos - 1, i_pos + 3});
                    move_bulge(H, v, s[2 * i_bulge], s[2 * i_bulge + 1]);

                    auto t1 = conj(v[0]);
                    auto v2 = v[1];
                    auto t2 = t1 * v2;
                    auto v3 = v[2];
                    auto t3 = t1 * v[2];
                    // Apply the reflector we just calculated from the right (but leave the last row for later)
                    for (idx_t j = 0; j < i_pos + 3; ++j)
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
                    if (i_pos > 0)
                    {
                        if (A(i_pos, i_pos - 1) != zero)
                        {
                            auto tst1 = abs1(A(i_pos - 1, i_pos - 1)) + abs1(A(i_pos, i_pos));
                            if (tst1 == zero)
                            {
                                if (i_pos > 1)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 2));
                                if (i_pos > 2)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 3));
                                if (i_pos > 3)
                                    tst1 += abs1(A(i_pos - 1, i_pos - 4));
                                if (i_pos < n - 1)
                                    tst1 += abs1(A(i_pos + 1, i_pos));
                                if (i_pos < n - 2)
                                    tst1 += abs1(A(i_pos + 2, i_pos));
                                if (i_pos < n - 3)
                                    tst1 += abs1(A(i_pos + 3, i_pos));
                            }
                            if (abs1(A(i_pos, i_pos - 1)) < max(small_num, eps * tst1))
                            {
                                auto ab = max(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto ba = min(abs1(A(i_pos, i_pos - 1)), abs1(A(i_pos - 1, i_pos)));
                                auto aa = max(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto bb = min(abs1(A(i_pos, i_pos)), abs1(A(i_pos, i_pos) - A(i_pos - 1, i_pos - 1)));
                                auto s = aa + ab;
                                if (ba * (ab / s) <= max(small_num, eps * (bb * (aa / s))))
                                {
                                    A(i_pos, i_pos - 1) = zero;
                                }
                            }
                        }
                    }
                }
            }

            i_bulge_start = (i_pos_last + 4 > n) ? (i_pos_last + 4 - n) / 2 : 0;

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
                for (idx_t j = i_pos + 1; j < n; ++j)
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
                idx_t i1 = (i_pos - 1) - (i_pos_last - 1 - ns + 2);
                idx_t i2 = min(nrows(Q), (i_pos_last - 1) + (i_pos_last - 1 - ns + 2) + 3);
                for (idx_t j = i1; j < i2; ++j)
                {
                    auto sum = Q(j, i_pos - 1) + v[1] * Q(j, i_pos) + v[2] * Q(j, i_pos + 1);
                    Q(j, i_pos - 1) = Q(j, i_pos - 1) - sum * v[0];
                    Q(j, i_pos) = Q(j, i_pos) - sum * v[0] * conj(v[1]);
                    Q(j, i_pos + 1) = Q(j, i_pos + 1) - sum * v[0] * conj(v[2]);
                }
            }
        }

        return;
    }

} // lapack

#endif // __RECURSIVE_BULGE_MOVE_HH__
