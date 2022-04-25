/// @file schur_swap.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlaexc.f
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __SCHUR_SWAP_HH__
#define __SCHUR_SWAP_HH__

#include <complex>
#include <cmath>
#include <assert.h>

#include <iostream>
#include <iomanip>

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "legacy_api/blas/utils.hpp"

namespace lapack
{

    /** schur_swap, swaps 2 eigenvalues of A.
     * 
     * @param[in]     want_q bool
     *                Whether or not to apply the transformations to Q
     * @param[in,out] A n-by-n matrix.
     *                Must be in Schur form
     * @param[in,out] Q n-by-n matrix.
     *                Orthogonal matrix, not referenced if want_q is false
     * @param[in]     j0 integer
     *                Index of first eigenvalue block
     * @param[in]     n1 integer
     *                Size of first eigenvalue block
     * @param[in]     n2 integer
     *                Size of second eigenvalue block
     *
     * @ingroup auxiliary
     */
    template <
        typename matrix_t,
        typename idx_t = size_type<matrix_t>>
    int schur_swap(bool want_q, matrix_t &A, matrix_t &Q, const idx_t &j0, const idx_t &n1, const idx_t &n2)
    {
        using T = type_t<matrix_t>;
        using blas::rot;
        using blas::rotg;
        using pair = std::pair<idx_t, idx_t>;
        using std::max;

        const idx_t n = ncols(A);
        const T zero(0);
        const T one(1);
        const T ten(10);

        assert(nrows(A) == n);
        assert(nrows(Q) == n);
        assert(ncols(Q) == n);
        assert(0 <= j0 and j0 < n);
        assert(n1 == 1 or n1 == 2);
        assert(n2 == 1 or n2 == 2);

        const idx_t j1 = j0 + 1;
        const idx_t j2 = j0 + 2;
        const idx_t j3 = j0 + 3;

        if (n1 == 1 and n2 == 1)
        {
            //
            // Swap two 1-by-1 blocks.
            //
            auto t00 = A(j0, j0);
            auto t11 = A(j1, j1);
            //
            // Determine the transformation to perform the interchange
            //
            T cs, sn;
            auto temp = A(j0, j1);
            auto temp2 = t11 - t00;
            rotg(temp, temp2, cs, sn);

            A(j1, j1) = t00;
            A(j0, j0) = t11;

            // Apply transformation from the left
            if (j2 < n)
            {
                auto row1 = slice(A, j0, pair{j2, n});
                auto row2 = slice(A, j1, pair{j2, n});
                rot(row1, row2, cs, sn);
            }
            // Apply transformation from the right
            if (j0 > 0)
            {
                auto col1 = slice(A, pair{0, j0}, j0);
                auto col2 = slice(A, pair{0, j0}, j1);
                rot(col1, col2, cs, sn);
            }
            if (want_q)
            {
                auto row1 = col(Q, j0);
                auto row2 = col(Q, j1);
                rot(row1, row2, cs, sn);
            }
        }
        if (n1 == 1 and n2 == 2)
        {
            //
            // Swap 1-by-1 block with 2-by-2 block
            //

            std::unique_ptr<T[]> _B(new T[6]);
            auto B = blas::internal::colmajor_matrix<T>(&_B[0], 3, 2);
            B(0, 0) = A(j0, j1);
            B(1, 0) = A(j1, j1) - A(j0, j0);
            B(2, 0) = A(j2, j1);
            B(0, 1) = A(j0, j2);
            B(1, 1) = A(j1, j2);
            B(2, 1) = A(j2, j2) - A(j0, j0);

            // Make B upper triangular
            T tau1, tau2;
            auto v1 = slice(B, pair{0, 3}, 0);
            auto v2 = slice(B, pair{1, 3}, 1);
            larfg(v1, tau1);
            auto sum = B(0, 1) + v1[1] * B(1, 1) + v1[2] * B(2, 1);
            B(0, 1) = B(0, 1) - sum * tau1;
            B(1, 1) = B(1, 1) - sum * tau1 * v1[1];
            B(2, 1) = B(2, 1) - sum * tau1 * v1[2];
            larfg(v2, tau2);

            //
            // Apply reflections to A and Q
            //

            // Reflections from the left
            for (idx_t j = j0; j < n; ++j)
            {
                auto sum = A(j0, j) + v1[1] * A(j1, j) + v1[2] * A(j2, j);
                A(j0, j) = A(j0, j) - sum * tau1;
                A(j1, j) = A(j1, j) - sum * tau1 * v1[1];
                A(j2, j) = A(j2, j) - sum * tau1 * v1[2];

                sum = A(j1, j) + v2[1] * A(j2, j);
                A(j1, j) = A(j1, j) - sum * tau2;
                A(j2, j) = A(j2, j) - sum * tau2 * v2[1];
            }
            // Reflections from the right
            for (idx_t j = 0; j < j3; ++j)
            {
                auto sum = A(j, j0) + v1[1] * A(j, j1) + v1[2] * A(j, j2);
                A(j, j0) = A(j, j0) - sum * tau1;
                A(j, j1) = A(j, j1) - sum * tau1 * v1[1];
                A(j, j2) = A(j, j2) - sum * tau1 * v1[2];

                sum = A(j, j1) + v2[1] * A(j, j2);
                A(j, j1) = A(j, j1) - sum * tau2;
                A(j, j2) = A(j, j2) - sum * tau2 * v2[1];
            }

            if (want_q)
            {

                for (idx_t j = 0; j < n; ++j)
                {
                    auto sum = Q(j, j0) + v1[1] * Q(j, j1) + v1[2] * Q(j, j2);
                    Q(j, j0) = Q(j, j0) - sum * tau1;
                    Q(j, j1) = Q(j, j1) - sum * tau1 * v1[1];
                    Q(j, j2) = Q(j, j2) - sum * tau1 * v1[2];

                    sum = Q(j, j1) + v2[1] * Q(j, j2);
                    Q(j, j1) = Q(j, j1) - sum * tau2;
                    Q(j, j2) = Q(j, j2) - sum * tau2 * v2[1];
                }
            }

            A(j2, j0) = zero;
            A(j2, j1) = zero;
        }
        if (n1 == 2 and n2 == 1)
        {
            //
            // Swap 2-by-2 block with 1-by-1 block
            //

            std::unique_ptr<T[]> _B(new T[6]);
            auto B = blas::internal::colmajor_matrix<T>(&_B[0], 3, 2);
            B(0, 0) = A(j1, j2);
            B(1, 0) = A(j1, j1) - A(j2, j2);
            B(2, 0) = A(j1, j0);
            B(0, 1) = A(j0, j2);
            B(1, 1) = A(j0, j1);
            B(2, 1) = A(j0, j0) - A(j2, j2);

            // Make B upper triangular
            T tau1, tau2;
            auto v1 = slice(B, pair{0, 3}, 0);
            auto v2 = slice(B, pair{1, 3}, 1);
            larfg(v1, tau1);
            auto sum = B(0, 1) + v1[1] * B(1, 1) + v1[2] * B(2, 1);
            B(0, 1) = B(0, 1) - sum * tau1;
            B(1, 1) = B(1, 1) - sum * tau1 * v1[1];
            B(2, 1) = B(2, 1) - sum * tau1 * v1[2];
            larfg(v2, tau2);

            //
            // Apply reflections to A and Q
            //

            // Reflections from the left
            for (idx_t j = j0; j < n; ++j)
            {
                auto sum = A(j2, j) + v1[1] * A(j1, j) + v1[2] * A(j0, j);
                A(j2, j) = A(j2, j) - sum * tau1;
                A(j1, j) = A(j1, j) - sum * tau1 * v1[1];
                A(j0, j) = A(j0, j) - sum * tau1 * v1[2];

                sum = A(j1, j) + v2[1] * A(j0, j);
                A(j1, j) = A(j1, j) - sum * tau2;
                A(j0, j) = A(j0, j) - sum * tau2 * v2[1];
            }
            // Reflections from the right
            for (idx_t j = 0; j < j3; ++j)
            {
                auto sum = A(j, j2) + v1[1] * A(j, j1) + v1[2] * A(j, j0);
                A(j, j2) = A(j, j2) - sum * tau1;
                A(j, j1) = A(j, j1) - sum * tau1 * v1[1];
                A(j, j0) = A(j, j0) - sum * tau1 * v1[2];

                sum = A(j, j1) + v2[1] * A(j, j0);
                A(j, j1) = A(j, j1) - sum * tau2;
                A(j, j0) = A(j, j0) - sum * tau2 * v2[1];
            }

            if (want_q)
            {
                for (idx_t j = 0; j < n; ++j)
                {
                    auto sum = Q(j, j2) + v1[1] * Q(j, j1) + v1[2] * Q(j, j0);
                    Q(j, j2) = Q(j, j2) - sum * tau1;
                    Q(j, j1) = Q(j, j1) - sum * tau1 * v1[1];
                    Q(j, j0) = Q(j, j0) - sum * tau1 * v1[2];

                    sum = Q(j, j1) + v2[1] * Q(j, j0);
                    Q(j, j1) = Q(j, j1) - sum * tau2;
                    Q(j, j0) = Q(j, j0) - sum * tau2 * v2[1];
                }
            }

            A(j1, j0) = zero;
            A(j2, j0) = zero;
        }
        if (n1 == 2 and n2 == 2)
        {
            std::unique_ptr<T[]> _D(new T[4 * 4]);
            auto D = blas::internal::colmajor_matrix<T>(&_D[0], 4, 4);

            auto AD_slice = slice(A, pair{j0, j0 + 4}, pair{j0, j0 + 4});
            lacpy(Uplo::General, AD_slice, D);
            auto dnorm = lange(Norm::Max, D);

            const T eps = blas::uroundoff<T>();
            const T small_num = blas::safe_min<T>() / eps;
            T thresh = max(ten * eps * dnorm, small_num);

            std::unique_ptr<T[]> _V(new T[4 * 2]);
            auto V = blas::internal::colmajor_matrix<T>(&_V[0], 4, 2);
            auto X = slice(V, pair{0, 2}, pair{0, 2});
            auto TL = slice(D, pair{0, 2}, pair{0, 2});
            auto TR = slice(D, pair{2, 4}, pair{2, 4});
            auto B = slice(D, pair{0, 2}, pair{2, 4});
            T scale, xnorm;
            lasy2(Op::NoTrans, Op::NoTrans, -1, TL, TR, B, scale, X, xnorm);

            V(2, 0) = -scale;
            V(2, 1) = zero;
            V(3, 0) = zero;
            V(3, 1) = -scale;

            // Make V upper triangular
            T tau1, tau2;
            auto v1 = slice(V, pair{0, 4}, 0);
            auto v2 = slice(V, pair{1, 4}, 1);
            larfg(v1, tau1);
            auto sum = V(0, 1) + v1[1] * V(1, 1) + v1[2] * V(2, 1) + v1[3] * V(3, 1);
            V(0, 1) = V(0, 1) - sum * tau1;
            V(1, 1) = V(1, 1) - sum * tau1 * v1[1];
            V(2, 1) = V(2, 1) - sum * tau1 * v1[2];
            V(3, 1) = V(3, 1) - sum * tau1 * v1[3];
            larfg(v2, tau2);

            // Apply reflections to D to check error
            for (idx_t j = 0; j < 4; ++j)
            {
                auto sum = D(0, j) + v1[1] * D(1, j) + v1[2] * D(2, j) + v1[3] * D(3, j);
                D(0, j) = D(0, j) - sum * tau1;
                D(1, j) = D(1, j) - sum * tau1 * v1[1];
                D(2, j) = D(2, j) - sum * tau1 * v1[2];
                D(3, j) = D(3, j) - sum * tau1 * v1[3];

                sum = D(1, j) + v2[1] * D(2, j) + v2[2] * D(3, j);
                D(1, j) = D(1, j) - sum * tau2;
                D(2, j) = D(2, j) - sum * tau2 * v2[1];
                D(3, j) = D(3, j) - sum * tau2 * v2[2];
            }
            for (idx_t j = 0; j < 4; ++j)
            {
                auto sum = D(j, 0) + v1[1] * D(j, 1) + v1[2] * D(j, 2) + v1[3] * D(j, 3);
                D(j, 0) = D(j, 0) - sum * tau1;
                D(j, 1) = D(j, 1) - sum * tau1 * v1[1];
                D(j, 2) = D(j, 2) - sum * tau1 * v1[2];
                D(j, 3) = D(j, 3) - sum * tau1 * v1[3];

                sum = D(j, 1) + v2[1] * D(j, 2) + v2[2] * D(j, 3);
                D(j, 1) = D(j, 1) - sum * tau2;
                D(j, 2) = D(j, 2) - sum * tau2 * v2[1];
                D(j, 3) = D(j, 3) - sum * tau2 * v2[2];
            }

            if (max(max(abs(D(2, 0)), abs(D(2, 1))), max(abs(D(3, 0)), abs(D(3, 1)))) > thresh)
                return 1;

            // Reflections from the left
            for (idx_t j = j0; j < n; ++j)
            {
                auto sum = A(j0, j) + v1[1] * A(j1, j) + v1[2] * A(j2, j) + v1[3] * A(j3, j);
                A(j0, j) = A(j0, j) - sum * tau1;
                A(j1, j) = A(j1, j) - sum * tau1 * v1[1];
                A(j2, j) = A(j2, j) - sum * tau1 * v1[2];
                A(j3, j) = A(j3, j) - sum * tau1 * v1[3];

                sum = A(j1, j) + v2[1] * A(j2, j) + v2[2] * A(j3, j);
                A(j1, j) = A(j1, j) - sum * tau2;
                A(j2, j) = A(j2, j) - sum * tau2 * v2[1];
                A(j3, j) = A(j3, j) - sum * tau2 * v2[2];
            }
            // Reflections from the right
            for (idx_t j = 0; j < j0 + 4; ++j)
            {
                auto sum = A(j, j0) + v1[1] * A(j, j1) + v1[2] * A(j, j2) + v1[3] * A(j, j3);
                A(j, j0) = A(j, j0) - sum * tau1;
                A(j, j1) = A(j, j1) - sum * tau1 * v1[1];
                A(j, j2) = A(j, j2) - sum * tau1 * v1[2];
                A(j, j3) = A(j, j3) - sum * tau1 * v1[3];

                sum = A(j, j1) + v2[1] * A(j, j2) + v2[2] * A(j, j3);
                A(j, j1) = A(j, j1) - sum * tau2;
                A(j, j2) = A(j, j2) - sum * tau2 * v2[1];
                A(j, j3) = A(j, j3) - sum * tau2 * v2[2];
            }

            if (want_q)
            {

                for (idx_t j = 0; j < n; ++j)
                {
                    auto sum = Q(j, j0) + v1[1] * Q(j, j1) + v1[2] * Q(j, j2) + v1[3] * Q(j, j3);
                    Q(j, j0) = Q(j, j0) - sum * tau1;
                    Q(j, j1) = Q(j, j1) - sum * tau1 * v1[1];
                    Q(j, j2) = Q(j, j2) - sum * tau1 * v1[2];
                    Q(j, j3) = Q(j, j3) - sum * tau1 * v1[3];

                    sum = Q(j, j1) + v2[1] * Q(j, j2) + v2[2] * Q(j, j3);
                    Q(j, j1) = Q(j, j1) - sum * tau2;
                    Q(j, j2) = Q(j, j2) - sum * tau2 * v2[1];
                    Q(j, j3) = Q(j, j3) - sum * tau2 * v2[2];
                }
            }

            A(j2, j0) = zero;
            A(j2, j1) = zero;
            A(j3, j0) = zero;
            A(j3, j1) = zero;
        }

        // Standardize the 2x2 Schur blocks (if any)
        if (n2 == 2)
        {
            T cs, sn;
            std::complex<T> s1, s2;
            lahqr_schur22(A(j0, j0), A(j0, j1), A(j1, j0), A(j1, j1), s1, s2, cs, sn); // Apply transformation from the left
            if (j2 < n)
            {
                auto row1 = slice(A, j0, pair{j2, n});
                auto row2 = slice(A, j1, pair{j2, n});
                rot(row1, row2, cs, sn);
            }
            // Apply transformation from the right
            if (j0 > 0)
            {
                auto col1 = slice(A, pair{0, j0}, j0);
                auto col2 = slice(A, pair{0, j0}, j1);
                rot(col1, col2, cs, sn);
            }
            if (want_q)
            {
                auto row1 = col(Q, j0);
                auto row2 = col(Q, j1);
                rot(row1, row2, cs, sn);
            }
        }
        if (n1 == 2)
        {
            idx_t j0_2 = j0 + n2;
            idx_t j1_2 = j0_2 + 1;

            T cs, sn;
            std::complex<T> s1, s2;
            lahqr_schur22(A(j0_2, j0_2), A(j0_2, j1_2), A(j1_2, j0_2), A(j1_2, j1_2), s1, s2, cs, sn); // Apply transformation from the left
            if (j2 < n)
            {
                auto row1 = slice(A, j0_2, pair{j0_2+2, n});
                auto row2 = slice(A, j1_2, pair{j0_2+2, n});
                rot(row1, row2, cs, sn);
            }
            // Apply transformation from the right
            if (j0_2 > 0)
            {
                auto col1 = slice(A, pair{0, j0_2}, j0_2);
                auto col2 = slice(A, pair{0, j0_2}, j1_2);
                rot(col1, col2, cs, sn);
            }
            if (want_q)
            {
                auto row1 = col(Q, j0_2);
                auto row2 = col(Q, j1_2);
                rot(row1, row2, cs, sn);
            }
        }

        return 0;
    }

} // lapack

#endif // __SCHUR_SWAP_HH__