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
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/scal.hpp"
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
template <TLAPACK_SMATRIX A_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX X_t,
          TLAPACK_SMATRIX matrixY_t>
int labrd(A_t& A, vector_t& tauq, vector_t& taup, X_t& X, matrixY_t& Y)
{
    using TA = type_t<A_t>;
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;
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
            if (i > 0) {
                auto y = slice(Y, i, range{0, i});
                auto A2 = slice(A, range{i, m}, range{0, i});
                auto a21 = slice(A, range{i, m}, i);
                conjugate(y);
                gemv(NO_TRANS, -one, A2, y, one, a21);
                conjugate(y);

                auto X2 = slice(X, range{i, m}, range{0, i});
                auto a22 = slice(A, range{0, i}, i);
                real_t e = real(A(i - 1, i));
                A(i - 1, i) = one;
                gemv(NO_TRANS, -one, X2, a22, one, a21);
                A(i - 1, i) = e;
            }

            // Generate reflection Q(i) to annihilate A(i+1:m,i)
            auto v = slice(A, range{i, m}, i);
            larfg(FORWARD, COLUMNWISE_STORAGE, v, tauq[i]);

            if (i < n - 1) {
                real_t d = real(A(i, i));
                A(i, i) = one;

                //
                // Compute Y(i+1:n,i) = y11
                //
                {
                    auto y11 = slice(Y, range{i + 1, n}, i);

                    // y11 = A(i:m,i+1:n)^H * v
                    auto A0 = slice(A, range{i, m}, range{i + 1, n});
                    gemv(CONJ_TRANS, one, A0, v, zero, y11);
                    // t = A(i:m,0:i)^H * v
                    auto A1 = slice(A, range{i, m}, range{0, i});
                    auto t = slice(Y, range{0, i}, i);
                    gemv(CONJ_TRANS, one, A1, v, zero, t);
                    // y11 = y11 - Y(i+1:n,0:i) * t
                    auto Y2 = slice(Y, range{i + 1, n}, range{0, i});
                    gemv(NO_TRANS, -one, Y2, t, one, y11);
                    // t = X(i:m,0:i)^H * v
                    auto X2 = slice(X, range{i, m}, range{0, i});
                    gemv(CONJ_TRANS, one, X2, v, zero, t);
                    // y11 = y11 - A(0:i,i+1:n)^H * t
                    auto A2 = slice(A, range{0, i}, range{i + 1, n});
                    gemv(CONJ_TRANS, -one, A2, t, one, y11);
                    // y11 = y11 * tauq(i)
                    scal(tauq[i], y11);
                }

                //
                // Update A(i,i+1:n) = a11
                //
                {
                    auto a11 = slice(A, i, range{i + 1, n});

                    // a11 = conj(a11) - Y(i+1:n,0:i+1)*conj(s)
                    // for (idx_t l = 0; l < n; ++l)
                    //     A(i, l) = conj(A(i, l));

                    auto s = slice(A, i, range{0, i + 1});
                    conjugate(a11);
                    conjugate(s);
                    auto Y3 = slice(Y, range{i + 1, n}, range{0, i + 1});
                    gemv(NO_TRANS, -one, Y3, s, one, a11);
                    conjugate(s);
                    // a11 = a11 - A(0:i,i+1:n)^H * conj(X(i,0:i))
                    auto A3 = slice(A, range{0, i}, range{i + 1, n});
                    auto x = slice(X, i, range{0, i});
                    conjugate(x);
                    gemv(CONJ_TRANS, -one, A3, x, one, a11);
                    conjugate(x);
                }

                //
                // Generate reflection P(i) to annihilate A(i,i+2:n)
                //
                auto w = slice(A, i, range{i + 1, n});
                larfg(FORWARD, COLUMNWISE_STORAGE, w, taup[i]);
                real_t e = real(A(i, i + 1));
                A(i, i + 1) = one;

                //
                // Compute X(i+1:m,i) = x11
                //
                {
                    auto x11 = slice(X, range{i + 1, m}, i);

                    // x11 = A(i+1:m,i+1:n) * w
                    auto A4 = slice(A, range{i + 1, m}, range{i + 1, n});
                    gemv(NO_TRANS, one, A4, w, zero, x11);
                    // t = Y(i+1:n,0:i+1)^H * w
                    auto Y4 = slice(Y, range{i + 1, n}, range{0, i + 1});
                    auto t2 = slice(X, range{0, i + 1}, i);
                    gemv(CONJ_TRANS, one, Y4, w, zero, t2);
                    // x11 = x11 - A(i+1:m,0:i+1) * t
                    auto A5 = slice(A, range{i + 1, m}, range{0, i + 1});
                    gemv(NO_TRANS, -one, A5, t2, one, x11);
                    // t = A(0:i,i+1:n) * w
                    auto A6 = slice(A, range{0, i}, range{i + 1, n});
                    auto t3 = slice(X, range{0, i}, i);
                    gemv(NO_TRANS, one, A6, w, zero, t3);
                    // x11 = x11 - X(i+1:m,0:i) * t
                    auto X4 = slice(X, range{i + 1, m}, range{0, i});
                    gemv(NO_TRANS, -one, X4, t3, one, x11);
                    // x11 = x11 * taup(i)
                    scal(taup[i], x11);
                    conjugate(w);
                }

                A(i, i) = d;
                A(i, i + 1) = e;
            }
        }
    }
    else {
        //
        // Reduce to lower bidiagonal form
        //
        for (idx_t i = 0; i < nb; ++i) {
            auto w = slice(A, i, range{i, n});
            conjugate(w);

            // Update A(i,i:n)
            if (i > 0) {
                auto Y2 = slice(Y, range{i, n}, range{0, i});
                auto s = slice(A, i, range{0, i});
                conjugate(s);
                real_t e = real(A(i, i - 1));
                A(i, i - 1) = one;
                gemv(NO_TRANS, -one, Y2, s, one, w);
                A(i, i - 1) = e;
                conjugate(s);

                auto A2 = slice(A, range{0, i}, range{i, n});
                auto x = slice(X, i, range{0, i});
                conjugate(x);
                gemv(CONJ_TRANS, -one, A2, x, one, w);
                conjugate(x);
            }

            // Generate reflection P(i) to annihilate A(i,i+1:n)
            larfg(FORWARD, COLUMNWISE_STORAGE, w, taup[i]);

            if (i + 1 < m) {
                real_t d = real(A(i, i));
                A(i, i) = one;

                //
                // Compute X(i+1:m,i) = x11
                //
                {
                    auto x11 = slice(X, range{i + 1, m}, i);

                    // x11 = A(i+1:m,i+1:n) * w
                    auto A4 = slice(A, range{i + 1, m}, range{i, n});
                    gemv(NO_TRANS, one, A4, w, zero, x11);
                    if (i > 0) {
                        // t = Y(i:n,0:i)^H * w
                        auto Y4 = slice(Y, range{i, n}, range{0, i});
                        auto t2 = slice(X, range{0, i}, i);
                        gemv(CONJ_TRANS, one, Y4, w, zero, t2);
                        // x11 = x11 - A(i+1:m,0:i) * t
                        auto A5 = slice(A, range{i + 1, m}, range{0, i});
                        gemv(NO_TRANS, -one, A5, t2, one, x11);
                        // t = A(0:i,i:n) * w
                        auto A6 = slice(A, range{0, i}, range{i, n});
                        auto t3 = slice(X, range{0, i}, i);
                        gemv(NO_TRANS, one, A6, w, zero, t3);
                        // x11 = x11 - X(i+1:m,0:i) * t
                        auto X4 = slice(X, range{i + 1, m}, range{0, i});
                        gemv(NO_TRANS, -one, X4, t3, one, x11);
                    }
                    // x11 = x11 * taup(i)
                    scal(taup[i], x11);
                    conjugate(w);
                }

                // Update A(i+1:m,i)
                {
                    auto y = slice(Y, i, range{0, i});
                    auto A2 = slice(A, range{i + 1, m}, range{0, i});
                    auto a21 = slice(A, range{i + 1, m}, i);
                    conjugate(y);
                    gemv(NO_TRANS, -one, A2, y, one, a21);
                    conjugate(y);

                    auto X2 = slice(X, range{i + 1, m}, range{0, i + 1});
                    auto a22 = slice(A, range{0, i + 1}, i);
                    gemv(NO_TRANS, -one, X2, a22, one, a21);
                }

                //
                // Generate reflection Q(i) to annihilate A(i+2:m,i)
                //
                auto v = slice(A, range{i + 1, m}, i);
                larfg(FORWARD, COLUMNWISE_STORAGE, v, tauq[i]);
                real_t e = real(A(i + 1, i));
                A(i + 1, i) = one;

                //
                // Compute Y(i+1:n,i) = y11
                //
                {
                    auto y11 = slice(Y, range{i + 1, n}, i);

                    // y11 = A(i+1:m,i+1:n)^H * v
                    auto A0 = slice(A, range{i + 1, m}, range{i + 1, n});
                    gemv(CONJ_TRANS, one, A0, v, zero, y11);
                    // t = A(i+1:m,0:i)^H * v
                    auto A1 = slice(A, range{i + 1, m}, range{0, i});
                    auto t = slice(Y, range{0, i}, i);
                    gemv(CONJ_TRANS, one, A1, v, zero, t);
                    // y11 = y11 - Y(i+1:n,0:i) * t
                    auto Y2 = slice(Y, range{i + 1, n}, range{0, i});
                    gemv(NO_TRANS, -one, Y2, t, one, y11);
                    // t = X(i+1:m,0:i+1)^H * v
                    auto t2 = slice(Y, range{0, i + 1}, i);
                    auto X3 = slice(X, range{i + 1, m}, range{0, i + 1});
                    gemv(CONJ_TRANS, one, X3, v, zero, t2);
                    // y11 = y11 - A(0:i+1,i+1:n)^H * t
                    auto A3 = slice(A, range{0, i + 1}, range{i + 1, n});
                    gemv(CONJ_TRANS, -one, A3, t2, one, y11);
                    // y11 = y11 * tauq(i)
                    scal(tauq[i], y11);
                }

                A(i, i) = d;
                A(i + 1, i) = e;
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
