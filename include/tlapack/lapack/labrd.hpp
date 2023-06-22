/// @file labrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zlabrd.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LABRD_HH
#define TLAPACK_LABRD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/blas/trmv.hpp"
#include "tlapack/lapack/conjugate.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Reduces the first nb rows and columns of a general
 * m by n matrix A to upper or lower bidiagonal form by an unitary
 * transformation Q**H * A * P, and returns the matrices X and Y which
 * are needed to apply the transformation to the unreduced part of A.
 *
 * If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
 * bidiagonal form.
 *
 * This is an auxiliary routine called by gebrd
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *
 * @param[out] d real vector of length nb.
 *      The diagonal elements of the bidiagonal matrix B.
 *
 * @param[out] e real vector of length nb.
 *      The off-diagonal elements of the bidiagonal matrix B.
 *
 * @param[out] tauq vector of length nb.
 *      The scalar factors of the elementary reflectors which represent the
 *      unitary matrix Q.
 *
 * @param[out] taup vector of length nb.
 *      The scalar factors of the elementary reflectors which represent the
 *      unitary matrix P.
 *
 * @param[out] X m-by-nb matrix.
 *
 * @param[out] Y n-by-nb matrix.
 *
 * @ingroup auxiliary
 */
template <class A_t,
          class vector_t,
          class r_vector_t,
          class X_t,
          class matrixY_t>
int labrd(A_t& A,
          r_vector_t& d,
          r_vector_t& e,
          vector_t& tauq,
          vector_t& taup,
          X_t& X,
          matrixY_t& Y)
{
    using TA = type_t<A_t>;
    using idx_t = size_type<A_t>;
    using pair = pair<idx_t, idx_t>;
    using real_t = real_type<TA>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t nb = ncols(X);

    // quick return if possible
    if (m == 0 or n == 0) return 0;

    if (m >= n) {
        //
        // Reduce to upper bidiagonal form
        //
        for (idx_t i = 0; i < nb; ++i) {
            // Update A(i:m,i)
            auto y = slice(Y, i, pair{0, i});
            auto A2 = slice(A, pair{i, m}, pair{0, i});
            auto a21 = slice(A, pair{i, m}, i);
            conjugate(y);
            gemv(noTranspose, -one, A2, y, one, a21);
            conjugate(y);

            auto X2 = slice(X, pair{i, m}, pair{0, i});
            auto a22 = slice(A, pair{0, i}, i);
            auto a23 = slice(A, pair{i, m}, i);
            gemv(noTranspose, -one, X2, a22, one, a23);
            // Generate reflection Q(i) to annihilate A(i+1:m,i)
            auto v = slice(A, pair{i, m}, i);
            larfg(forward, columnwise_storage, v, tauq[i]);
            d[i] = real(A(i, i));

            if (i < n - 1) {
                A(i, i) = one;
                //
                // Compute Y(i+1:n,i) = y11
                //
                auto y11 = slice(Y, pair{i + 1, n}, i);

                // y11 = A(i:m,i+1:n)^H * v
                auto A0 = slice(A, pair{i, m}, pair{i + 1, n});
                gemv(conjTranspose, one, A0, v, zero, y11);
                // t = A(i:m,0:i)^H * v
                auto A1 = slice(A, pair{i, m}, pair{0, i});
                auto t = slice(Y, pair{0, i}, i);
                gemv(conjTranspose, one, A1, v, zero, t);
                // y11 = y11 - Y(i+1:n,0:i) * t
                auto Y2 = slice(Y, pair{i + 1, n}, pair{0, i});
                gemv(noTranspose, -one, Y2, t, one, y11);
                // t = X(i:m,0:i)^H * v
                auto X2 = slice(X, pair{i, m}, pair{0, i});
                gemv(conjTranspose, one, X2, v, zero, t);
                // y11 = y11 - A(0:i,i+1:n)^H * t
                auto A2 = slice(A, pair{0, i}, pair{i + 1, n});
                gemv(conjTranspose, -one, A2, t, one, y11);
                // y11 = y11 * tauq(i)
                scal(tauq[i], y11);

                //
                // Update A(i,i+1:n) = a11
                //
                auto a11 = slice(A, i, pair{i + 1, n});

                // a11 = conj(a11) - Y(i+1:n,0:i+1)*conj(s)
                // for (idx_t l = 0; l < n; ++l)
                //     A(i, l) = conj(A(i, l));

                auto s = slice(A, i, pair{0, i + 1});
                conjugate(a11);
                conjugate(s);
                auto Y3 = slice(Y, pair{i + 1, n}, pair{0, i + 1});
                gemv(noTranspose, -one, Y3, s, one, a11);
                conjugate(s);
                // a11 = a11 - A(0:i,i+1:n)^H * conj(X(i,0:i))
                auto A3 = slice(A, pair{0, i}, pair{i + 1, n});
                auto x = slice(X, i, pair{0, i});
                conjugate(x);
                gemv(conjTranspose, -one, A3, x, one, a11);
                conjugate(x);

                //
                // Generate reflection P(i) to annihilate A(i,i+2:n)
                //
                auto w = slice(A, i, pair{i + 1, n});
                larfg(forward, columnwise_storage, w, taup[i]);
                e[i] = real(A(i, i + 1));
                A(i, i + 1) = one;

                //
                // Compute X(i+1:m,i) = x11
                //
                auto x11 = slice(X, pair{i + 1, m}, i);

                // x11 = A(i+1:m,i+1:n) * w
                auto A4 = slice(A, pair{i + 1, m}, pair{i + 1, n});
                gemv(noTranspose, one, A4, w, zero, x11);
                // t = Y(i+1:n,0:i+1)^H * w
                auto Y4 = slice(Y, pair{i + 1, n}, pair{0, i + 1});
                auto t2 = slice(X, pair{0, i + 1}, i);
                gemv(conjTranspose, one, Y4, w, zero, t2);
                // x11 = x11 - A(i+1:m,0:i+1) * t
                auto A5 = slice(A, pair{i + 1, m}, pair{0, i + 1});
                gemv(noTranspose, -one, A5, t2, one, x11);
                // t = A(0:i,i+1:n) * w
                auto A6 = slice(A, pair{0, i}, pair{i + 1, n});
                auto t3 = slice(X, pair{0, i}, i);
                gemv(noTranspose, one, A6, w, zero, t3);
                // x11 = x11 - X(i+1:m,0:i) * t
                auto X4 = slice(X, pair{i + 1, m}, pair{0, i});
                gemv(noTranspose, -one, X4, t3, one, x11);
                // x11 = x11 * taup(i)
                scal(taup[i], x11);
                conjugate(w);
            }
        }
    }
    else {
        //
        // Reduce to lower bidiagonal form
        //
        for (idx_t i = 0; i < nb; ++i) {
            // Update A(i,i:n)
            auto Y2 = slice(Y, pair{i, n}, pair{0, i});
            auto s = slice(A, i, pair{0, i});
            auto a12 = slice(A, i, pair{i, n});
            conjugate(s);
            conjugate(a12);
            gemv(noTranspose, -one, Y2, s, one, a12);
            conjugate(s);

            auto A2 = slice(A, pair{0, i}, pair{i, n});
            auto x = slice(X, i, pair{0, i});
            conjugate(x);
            gemv(conjTranspose, -one, A2, x, one, a12);
            conjugate(x);
            //
            // Generate reflection P(i) to annihilate A(i,i+1:n)
            //
            //
            auto w = slice(A, i, pair{i, n});
            larfg(forward, columnwise_storage, w, taup[i]);
            d[i] = real(A(i, i));
            A(i, i) = one;

            if (i + 1 < m) {
                //
                // Compute X(i+1:m,i) = x11
                //
                auto x11 = slice(X, pair{i + 1, m}, i);

                // x11 = A(i+1:m,i+1:n) * w
                auto A4 = slice(A, pair{i + 1, m}, pair{i, n});
                gemv(noTranspose, one, A4, w, zero, x11);
                if (i > 0) {
                    // t = Y(i:n,0:i)^H * w
                    auto Y4 = slice(Y, pair{i, n}, pair{0, i});
                    auto t2 = slice(X, pair{0, i}, i);
                    gemv(conjTranspose, one, Y4, w, zero, t2);
                    // x11 = x11 - A(i+1:m,0:i) * t
                    auto A5 = slice(A, pair{i + 1, m}, pair{0, i});
                    gemv(noTranspose, -one, A5, t2, one, x11);
                    // t = A(0:i,i:n) * w
                    auto A6 = slice(A, pair{0, i}, pair{i, n});
                    auto t3 = slice(X, pair{0, i}, i);
                    gemv(noTranspose, one, A6, w, zero, t3);
                    // x11 = x11 - X(i+1:m,0:i) * t
                    auto X4 = slice(X, pair{i + 1, m}, pair{0, i});
                    gemv(noTranspose, -one, X4, t3, one, x11);
                }
                // x11 = x11 * taup(i)
                scal(taup[i], x11);
                conjugate(w);

                // Update A(i+1:m,i)
                auto y = slice(Y, i, pair{0, i});
                auto A2 = slice(A, pair{i + 1, m}, pair{0, i});
                auto a21 = slice(A, pair{i + 1, m}, i);
                conjugate(y);
                gemv(noTranspose, -one, A2, y, one, a21);
                conjugate(y);

                auto X2 = slice(X, pair{i + 1, m}, pair{0, i + 1});
                auto a22 = slice(A, pair{0, i + 1}, i);
                gemv(noTranspose, -one, X2, a22, one, a21);

                //
                // Generate reflection Q(i) to annihilate A(i+2:m,i)
                //
                auto v = slice(A, pair{i + 1, m}, i);
                larfg(forward, columnwise_storage, v, tauq[i]);
                e[i] = real(A(i + 1, i));
                A(i + 1, i) = one;

                //
                // Compute Y(i+1:n,i) = y11
                //
                auto y11 = slice(Y, pair{i + 1, n}, i);

                // y11 = A(i+1:m,i+1:n)^H * v
                auto A0 = slice(A, pair{i + 1, m}, pair{i + 1, n});
                gemv(conjTranspose, one, A0, v, zero, y11);
                // t = A(i+1:m,0:i)^H * v
                auto A1 = slice(A, pair{i + 1, m}, pair{0, i});
                auto t = slice(Y, pair{0, i}, i);
                gemv(conjTranspose, one, A1, v, zero, t);
                // y11 = y11 - Y(i+1:n,0:i) * t
                auto Y2 = slice(Y, pair{i + 1, n}, pair{0, i});
                gemv(noTranspose, -one, Y2, t, one, y11);
                // t = X(i+1:m,0:i+1)^H * v
                auto t2 = slice(Y, pair{0, i + 1}, i);
                auto X3 = slice(X, pair{i + 1, m}, pair{0, i + 1});
                gemv(conjTranspose, one, X3, v, zero, t2);
                // y11 = y11 - A(0:i+1,i+1:n)^H * t
                auto A3 = slice(A, pair{0, i + 1}, pair{i + 1, n});
                gemv(conjTranspose, -one, A3, t2, one, y11);
                // y11 = y11 * tauq(i)
                scal(tauq[i], y11);
            }
            else {
                conjugate(w);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LABRD_HH
