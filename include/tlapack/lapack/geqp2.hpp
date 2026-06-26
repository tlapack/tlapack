/// @file geqp2.hpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
// Test utilities and definitions (must come before <T>LAPACK headers)

#ifndef TLAPACK_GEQP2_HH
#define TLAPACK_GEQP2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Computes a QR factorization with column pivoting of a matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * Column pivoting chooses at each step the column with largest 2-norm,
 * introducing a permutation matrix P such that:
 * \[
 *          A * P = Q * R
 * \]
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in,out] p Index vector of length n.
 *      On exit, p[i] = j means that the i-th column of the permuted matrix
 *      was originally the j-th column of the input matrix A.
 *
 * @param[in,out] vn1 Real vector of length n.
 *      On entry and exit, vn1[j] contains the norm of the vector A(j:m,j) for
 *      j = 1, n. Used for tracking column norms during the QR factorization
 * with pivoting.
 *
 *
 * @param[in,out] vn2 Real vector of length n.
 *      On entry and exit, vn2[j] contains the norm of the vector A(j:m,j) for
 *      j = 1, n. Used as the base norm for updating vn1  during the
 * factorization.
 *
 * @ingroup computational
 */

template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vectorTau_t,
          TLAPACK_VECTOR idx_vector_t,
          TLAPACK_VECTOR vectorVN1_t,
          TLAPACK_VECTOR vectorVN2_t>
int geqp2(matrix_t& A,
          vectorTau_t& tau,
          idx_vector_t& p,
          vectorVN1_t& vn1,
          vectorVN2_t& vn2)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const real_t eps = ulp<real_t>();
    const real_t safety_threshold = sqrt(eps);

    // Create matrix views of the vectors p, vn1, and vn2
    LegacyMatrix<idx_t> P(1, n, &p[0], 1);
    LegacyMatrix<real_t> VN1(1, n, &vn1[0], 1);
    LegacyMatrix<real_t> VN2(1, n, &vn2[0], 1);

    // Initialize column norms for the active block
    for (idx_t j = 0; j < n; ++j) {
        auto col = slice(A, range{0, m}, j);
        vn1[j] = nrm2(col);
        vn2[j] = vn1[j];
    }

    // Main Computational Loop
    for (idx_t k = 0; k < std::min(m, n); ++k) {
        // Identify pivot column from non-resolved active columns
        idx_t pivot = k;
        real_t max_norm = vn1[k];
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > max_norm) {
                max_norm = vn1[j];
                pivot = j;
            }
        }

        // Perform swap if necessary
        if (pivot != k) {
            auto current_col = col(A, k);
            auto next_col = col(A, pivot);
            swap<decltype(current_col), decltype(next_col)>(current_col,
                                                            next_col);

            auto k_col1 = col(VN1, k);
            auto k_col2 = col(VN2, k);
            auto p1_col = col(P, k);
            auto pivot_col1 = col(VN1, pivot);
            auto pivot_col2 = col(VN2, pivot);
            auto p2_col = col(P, pivot);

            swap<decltype(k_col1), decltype(pivot_col1)>(k_col1, pivot_col1);
            swap<decltype(k_col2), decltype(pivot_col2)>(k_col2, pivot_col2);
            swap<decltype(p1_col), decltype(p2_col)>(p1_col, p2_col);
        }

        // Generate Householder reflector for the k-th column
        auto col_k = slice(A, range{k, m}, k);
        larfg(FORWARD, COLUMNWISE_STORAGE, col_k, tau[k]);

        // Apply the reflector to the remaining columns
        if (k < n - 1) {
            auto A_trailing = slice(A, range{k, m}, range{k + 1, n});
            larf(LEFT_SIDE, FORWARD, COLUMNWISE_STORAGE, col_k, conj(tau[k]),
                 A_trailing);
        }

        // Update the norms of the remaining columns
        for (idx_t j = k + 1; j < n; ++j) {
            if (vn1[j] > static_cast<real_t>(0.0)) {
                // t = |A(k,j)| / vn1[j]
                real_t t = abs(A(k, j)) / vn1[j];
                // t = max(0, 1 - t^2)
                t = std::max(static_cast<real_t>(0.0),
                             (static_cast<real_t>(1.0) - t * t));
                // t2 = t * (vn1[j] / vn2[j])^2
                real_t norm_ratio = vn1[j] / vn2[j];
                real_t t2 = t * (norm_ratio * norm_ratio);
                // Safety check
                if (t2 <= safety_threshold) {
                    auto A_col = slice(A, range(k + 1, m), j);
                    vn1[j] = nrm2(A_col);
                    vn2[j] = vn1[j];  // Reset base
                }
                else {
                    vn1[j] *= std::sqrt(t);
                }
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEQP2_HH
