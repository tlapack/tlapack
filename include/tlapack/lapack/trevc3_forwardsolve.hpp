/// @file trevc3_forwardsolve.hpp
/// @author Thijs Steel, KU Leuven, Belgium
// based on A. Schwarz et al., "Scalable eigenvector computation for the
// non-symmetric eigenvalue problem"
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TREVC3_FORWARDSOLVE_HH
#define TLAPACK_TREVC3_FORWARDSOLVE_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/asum.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/trevc_forwardsolve.hpp"
#include "tlapack/lapack/trevc_protect.hpp"

namespace tlapack {

/**
 * Calculate the ks-th through ke-th (not inclusive) left eigenvector of T
 * using a blocked backsubstitution.
 *
 * @param[in] T      n-by-n matrix
 *                   Upper quasi-triangular matrix whose eigenvectors are to
 *                   be computed. The matrix is assumed (without checking) to be
 *                   in standardized Schur form. This mostly affects the 2x2
 *                   blocks for complex conjugate eigenvalue pairs. Where we
 *                   assume that the 2x2 blocks are of the form [ a  b; c  a ]
 *                   with b and c having opposite signs.
 *
 * @param[out] X     n-by-m matrix
 *                   On output, contains the ks-th through ke-th (not inclusive)
 *                   left eigenvectors of T, stored in the columns of X.
 *
 * @param[out] rwork Real workspace vector of size at least
 *                   min(blocksize + (ke - ks),2*(ke - ks))
 *
 * @param[out] work  Workspace vector of size at least (ke - ks)
 *
 * @param[in] ks     integer
 *
 * @param[in] ke     integer
 *                   ks and ke determine the range of eigenvectors to compute.
 *
 * @param[in] blocksize  integer
 *                       The blocksize to use in the blocked backsubstitution.
 */
template <TLAPACK_MATRIX matrix_T_t,
          TLAPACK_MATRIX matrix_X_t,
          TLAPACK_WORKSPACE rwork_t,
          TLAPACK_WORKSPACE work_t>
void trevc3_forwardsolve(const matrix_T_t& T,
                         matrix_X_t& X,
                         rwork_t& rwork,
                         work_t& work,
                         size_type<matrix_T_t> ks,
                         size_type<matrix_T_t> ke,
                         size_type<matrix_T_t> blocksize)
{
    using idx_t = size_type<matrix_T_t>;
    using TT = type_t<matrix_T_t>;
    using real_t = real_type<TT>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(T);

    const real_t sf_max = safe_max<real_t>();
    const real_t sf_min = safe_min<real_t>();

    tlapack_check(ncols(T) == n);

    idx_t nk = ke - ks;
    auto [shifts, work2] = reshape(work, nk);

    laset(Uplo::General, TT(0), TT(0), X);

    // Step 1: Calculate the eigenvectors of the ks:ke submatrix
    // using trevc_forwardsolve_single or trevc_forwardsolve_double
    // and update the rest of X using a matrix-matrix multiplication
    {
        // Calculate the eigenvectors of the ks:ke block
        auto Tii = slice(T, range(ks, ke), range(ks, ke));
        auto X_ii = slice(X, range(ks, ke), range(0, nk));

        auto [colN_ii, rwork2] = reshape(rwork, nk);
        trevc_colnorms(Norm::One, Tii, colN_ii);

        for (idx_t k = 0; k < nk;) {
            bool pair = false;
            if constexpr (is_real<TT>) {
                if (k + 1 < nk) {
                    if (Tii(k + 1, k) != TT(0)) {
                        pair = true;
                    }
                }
            }

            if (pair) {
                if constexpr (is_real<TT>) {
                    TT alpha = Tii(k, k);
                    TT beta = Tii(k, k + 1);
                    TT gamma = Tii(k + 1, k);

                    // real part of eigenvalue
                    TT wr = alpha;
                    // imaginary part of eigenvalue
                    TT wi = sqrt(abs(beta)) * sqrt(abs(gamma));

                    shifts[k] = wr;
                    shifts[k + 1] = wi;
                    // todo: remember that k and k+1 form a pair somehow, maybe
                    // a bool array?

                    // Complex conjugate pair
                    auto x1 = col(X_ii, k);
                    auto x2 = col(X_ii, k + 1);
                    trevc_forwardsolve_double(Tii, x1, x2, k, colN_ii);
                }
                k += 2;
            }
            else {
                shifts[k] = Tii(k, k);
                // Real eigenvalue
                auto x1 = col(X_ii, k);
                trevc_forwardsolve_single(Tii, x1, k, colN_ii);
                k += 1;
            }
        }

        if (ke < n) {
            auto T_ij = slice(T, range(ks, ke), range(ke, n));
            auto X_i = slice(X, range(ke, n), range(0, nk));

            // Calculate scaling factors to avoid overflow in the matrix-matrix
            // multiplication below
            real_t t_one_norm = lange(Norm::One, T_ij);
            for (idx_t j = 0; j < nk; ++j) {
                idx_t imax = iamax(col(X_ii, j));
                real_t x_norm = abs(X_ii(imax, j));

                imax = iamax(col(X_i, j));
                real_t y_norm = abs(X_i(imax, j));

                real_t scale =
                    trevc_protectupdate(y_norm, t_one_norm, x_norm, sf_max);

                if (scale != real_t(1)) {
                    // Scale X(:, j)
                    for (idx_t i = ks; i < n; ++i) {
                        X(i, j) = scale * X(i, j);
                    }
                }
            }

            // This is where you might think to initialize X_i as -T_ij
            // But the multiplication below takes care of that because it also
            // takes the element 1 into account
            gemm(Op::ConjTrans, Op::NoTrans, TT(-1), T_ij, X_ii, TT(1), X_i);
        }
    }

    // Step 2: Now keep propagating the updates downwards, but keep the
    // individual shifts in mind
    for (idx_t iib = ke; iib < n;) {
        idx_t nb = min(blocksize, n - iib);
        // Start of the block
        idx_t bs = iib;
        // End of the block
        idx_t be = iib + nb;

        // Make sure we don't split 2x2 blocks
        if constexpr (is_real<TT>) {
            if (be < n) {
                // TODO: find a better way to check this so we don't
                // always access other blocks of T
                if (T(be, be - 1) != TT(0)) {
                    be += 1;
                    nb += 1;
                }
            }
        }

        auto T_ii = slice(T, range(bs, be), range(bs, be));
        auto X_ii = slice(X, range(bs, be), range(0, nk));

        auto [colN_ii, rwork2] = reshape(rwork, nb);
        trevc_colnorms(Norm::One, T_ii, colN_ii);
        auto [scale_ii, rwork3] = reshape(rwork2, nk);
        for (idx_t j = 0; j < nk; ++j) {
            scale_ii[j] = real_t(1);
        }

        for (idx_t k = 0; k < nk;) {
            bool pair = false;
            if constexpr (is_real<TT>) {
                if (k + 1 < nk) {
                    if (T(ks + k + 1, ks + k) != TT(0)) {
                        pair = true;
                    }
                }
            }

            if (pair) {
                TT wr = shifts[k];
                TT wi = shifts[k + 1];

                for (idx_t i = 0; i < nb;) {
                    bool is_2x2_block = false;
                    if (i + 1 < nb) {
                        if (T_ii(i + 1, i) != TT(0)) {
                            is_2x2_block = true;
                        }
                    }

                    if (is_2x2_block) {
                        // 2x2 block

                        for (idx_t j = 0; j < i; ++j) {
                            X_ii(i, k) -= T_ii(j, i) * X_ii(j, k);
                            X_ii(i, k + 1) -= T_ii(j, i) * X_ii(j, k + 1);
                            X_ii(i + 1, k) -= T_ii(j, i + 1) * X_ii(j, k);
                            X_ii(i + 1, k + 1) -=
                                T_ii(j, i + 1) * X_ii(j, k + 1);
                        }

                        // Solve the complex 2x2 system
                        // Using real arithmetic only with Cramer's rule

                        TT a11r = T_ii(i, i) - wr;
                        TT a11i = wi;
                        // a12 and a21 are switched to transpose the system
                        TT a12 = T_ii(i + 1, i);
                        TT a21 = T_ii(i, i + 1);
                        TT a22r = T_ii(i + 1, i + 1) - wr;
                        TT a22i = wi;

                        TT b1r = X_ii(i, k);
                        TT b1i = X_ii(i, k + 1);
                        TT b2r = X_ii(i + 1, k);
                        TT b2i = X_ii(i + 1, k + 1);

                        TT detr = a11r * a22r - a11i * a22i - a12 * a21;
                        TT deti = a11r * a22i + a11i * a22r;

                        TT denom = detr * detr + deti * deti;

                        TT c1r = a22r * b1r - a22i * b1i - a12 * b2r;
                        TT c1i = a22r * b1i + a22i * b1r - a12 * b2i;
                        TT x1r = (c1r * detr + c1i * deti) / denom;
                        TT x1i = (c1i * detr - c1r * deti) / denom;

                        TT c2r = (a11r * b2r - a11i * b2i) - (a21 * b1r);
                        TT c2i = (a11r * b2i + a11i * b2r) - (a21 * b1i);
                        TT x2r = (c2r * detr + c2i * deti) / denom;
                        TT x2i = (c2i * detr - c2r * deti) / denom;

                        X_ii(i, k) = x1r;
                        X_ii(i, k + 1) = x1i;
                        X_ii(i + 1, k) = x2r;
                        X_ii(i + 1, k + 1) = x2i;

                        i += 2;
                    }
                    else {
                        // 1x1 block
                        for (idx_t j = 0; j < i; ++j) {
                            X_ii(i, k) -= T_ii(j, i) * X_ii(j, k);
                            X_ii(i, k + 1) -= T_ii(j, i) * X_ii(j, k + 1);
                        }

                        // Do the complex division:
                        // (v1_r[i] + i*v1_i[i]) / (T11(i, i) - (wr + i*wi))
                        // in real arithmetic only
                        TT a = X_ii(i, k);
                        TT b = X_ii(i, k + 1);
                        TT c = T_ii(i, i) - wr;
                        TT d = wi;
                        TT denom = c * c + d * d;
                        X_ii(i, k) = (a * c + b * d) / denom;
                        X_ii(i, k + 1) = (b * c - a * d) / denom;

                        i += 1;
                    }
                }

                k += 2;
            }
            else {
                TT w = shifts[k];

                if constexpr (is_complex<TT>) {
                    // The matrix is complex, so there are no two-by-two blocks
                    // to consider

                    for (idx_t i = 0; i < nb; ++i) {
                        idx_t ixmax = iamax(slice(X_ii, range(0, i), k));
                        real_t xmax = abs1(X_ii(ixmax, k));

                        real_t tnorm = colN_ii[i];

                        real_t scale1 = trevc_protectupdate(
                            abs1(X_ii(i, k)), tnorm, xmax, sf_max);

                        if (scale1 != real_t(1)) {
                            // Scale the current part of the vector
                            for (idx_t jj = 0; jj < nb; ++jj) {
                                X_ii(jj, k) = scale1 * X_ii(jj, k);
                            }
                            scale_ii[k] *= scale1;
                        }

                        real_t scale2 = trevc_protectdiv(
                            X_ii(i, k), T_ii(i, i) - w, sf_min, sf_max);

                        if (scale2 != real_t(1)) {
                            // Scale the current part of the vector
                            for (idx_t jj = 0; jj < nb; ++jj) {
                                X_ii(jj, k) = scale2 * X_ii(jj, k);
                            }
                            scale_ii[k] *= scale2;
                        }

                        for (idx_t j = 0; j < i; ++j) {
                            X_ii(i, k) -= conj(T_ii(j, i)) * X_ii(j, k);
                        }

                        X_ii(i, k) = X_ii(i, k) / conj(T_ii(i, i) - w);
                    }
                }
                else {
                    // The matrix is real, so we need to consider potential
                    // 2x2 blocks
                    // The matrix is real, so we need to consider potential
                    // 2x2 blocks for complex conjugate eigenvalue pairs
                    idx_t i = 0;
                    while (i < nb) {
                        bool is_2x2_block = false;
                        if (i + 1 < nb) {
                            if (T_ii(i + 1, i) != TT(0)) {
                                is_2x2_block = true;
                            }
                        }

                        if (is_2x2_block) {
                            // 2x2 block

                            for (idx_t j = 0; j < i; ++j) {
                                X_ii(i, k) -= T_ii(j, i) * X_ii(j, k);
                                X_ii(i + 1, k) -= T_ii(j, i + 1) * X_ii(j, k);
                            }

                            // Solve the 2x2 (transposed) system:
                            // [T33(i,i)-w   T33(i+1,i)    ] [v3[i]  ] = [rhs1]
                            // [T33(i,i+1)   T33(i+1,i+1)-w] [v3[i+1]]   [rhs2]
                            TT rhs1 = X_ii(i, k);
                            TT rhs2 = X_ii(i + 1, k);

                            TT a = T_ii(i, i) - w;
                            TT b = T_ii(i + 1, i);
                            TT c = T_ii(i, i + 1);
                            TT d = T_ii(i + 1, i + 1) - w;

                            TT det = a * d - b * c;

                            X_ii(i, k) = (d * rhs1 - b * rhs2) / det;
                            X_ii(i + 1, k) = (-c * rhs1 + a * rhs2) / det;

                            i += 2;
                        }
                        else {
                            // 1x1 block

                            idx_t ixmax = iamax(slice(X_ii, range(0, i), k));
                            real_t xmax = abs(X_ii(ixmax, k));

                            real_t tnorm = colN_ii[i];

                            real_t scale1 = trevc_protectupdate(
                                abs1(X_ii(i, k)), tnorm, xmax, sf_max);

                            if (scale1 != real_t(1)) {
                                // Scale the current part of the vector
                                for (idx_t jj = 0; jj < nb; ++jj) {
                                    X_ii(jj, k) = scale1 * X_ii(jj, k);
                                }
                                scale_ii[k] *= scale1;
                            }

                            for (idx_t j = 0; j < i; ++j) {
                                X_ii(i, k) -= T_ii(j, i) * X_ii(j, k);
                            }

                            real_t scale2 = trevc_protectdiv(
                                X_ii(i, k), T_ii(i, i) - w, sf_min, sf_max);

                            if (scale2 != real_t(1)) {
                                // Scale the current part of the vector
                                for (idx_t jj = 0; jj < nb; ++jj) {
                                    X_ii(jj, k) = scale2 * X_ii(jj, k);
                                }
                                scale_ii[k] *= scale2;
                            }

                            X_ii(i, k) = X_ii(i, k) / (T_ii(i, i) - w);

                            i += 1;
                        }
                    }
                }

                k += 1;
            }
        }

        // @TODO: this can be optimized further by combining the two loops

        // @TODO: this scaling can also be delayed, requiring careful storage
        // of which block has already been scaled by what factor
        // See e.g. LAPACK's DLATRS3 implementation by Angelika Schwarz

        // Apply the scale factors to the rest of X
        for (idx_t j = 0; j < nk; ++j) {
            real_t scale = scale_ii[j];
            if (scale != real_t(1)) {
                for (idx_t i = be; i < n; ++i) {
                    X(i, j) = scale * X(i, j);
                }
                for (idx_t i = ks; i < bs; ++i) {
                    X(i, j) = scale * X(i, j);
                }
            }
        }

        if (be < n) {
            auto T_ij = slice(T, range(bs, be), range(be, n));
            auto X_i = slice(X, range(be, n), range(0, nk));

            // Calculate scaling factors to avoid overflow in the matrix-matrix
            // multiplication below
            real_t t_one_norm = lange(Norm::One, T_ij);
            for (idx_t j = 0; j < nk; ++j) {
                idx_t imax = iamax(col(X_ii, j));
                real_t x_norm = abs(X_ii(imax, j));

                imax = iamax(col(X_i, j));
                real_t y_norm = abs(X_i(imax, j));

                real_t scale =
                    trevc_protectupdate(y_norm, t_one_norm, x_norm, sf_max);

                if (scale != real_t(1)) {
                    // Scale X(:, j)
                    for (idx_t i = ks; i < n; ++i) {
                        X(i, j) = scale * X(i, j);
                    }
                }
            }

            gemm(Op::ConjTrans, Op::NoTrans, TT(-1), T_ij, X_ii, TT(1), X_i);
        }

        iib += nb;
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TREVC3_HH
