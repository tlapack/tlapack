/// @file geqpf.hpp
/// @author Racheal Asamoah, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAQPS_HH
#define TLAPACK_LAQPS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/plugins/stdvector.hpp"

namespace tlapack {

/** Worspace query of geqpf()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau Not referenced.
 *
 * @param[in] opts Options.
 *
 * @param[in,out] workinfo
 *      On output, the amount workspace required. It is larger than or equal
 *      to that given on input.
 *
 * @ingroup workspace_query
 */
template <class matrix_t, class vector_idx, class vector_t>
inline constexpr void laqps_worksize(const matrix_t& A,
                                     const vector_idx& jpvt,
                                     const vector_t& tau,
                                     workinfo_t& workinfo,
                                     const workspace_opts_t<>& opts = {})
{
    using idx_t = size_type<matrix_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto C = cols(A, range<idx_t>{1, n});
        larf_worksize(left_side, forward, col(A, 0), tau[0], C, workinfo, opts);
    }
}

template <class matrix_t, class vector_idx, class vector_t, class vector2_t>
int laqps(matrix_t& A,
          vector_idx& jpvt,
          vector_t& tau,
          vector2_t& partial_norms,
          vector2_t& exact_norms,
          const workspace_opts_t<>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // Functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = std::min<idx_t>(m, n);

    const idx_t nb_opts = 3;
    const idx_t nb = std::min<idx_t>(nb_opts, k);

    std::vector<T> auxv_;
    auto auxv = new_matrix(auxv_, nb, 1);

    std::vector<T> F_;
    auto F = new_matrix(F_, n, nb);

    const real_t one(1);
    const real_t zero(0);

    const real_t eps = ulp<real_t>();
    const real_t tol3z = sqrt(eps);

    for (idx_t i = 0; i < nb; ++i) {
        //
        //          Determine ith pivot column and swap if necessary
        //
        jpvt[i] = i;
        for (idx_t j = i + 1; j < n; j++) {
            if (partial_norms[j] > partial_norms[jpvt[i]]) jpvt[i] = j;
        }
        auto ai = col(A, i);
        auto bi = col(A, jpvt[i]);
        tlapack::swap(ai, bi);
        auto frow1 = row(F, i);
        auto frow2 = row(F, jpvt[i]);
        tlapack::swap(frow1, frow2);
        std::swap(partial_norms[i], partial_norms[jpvt[i]]);
        std::swap(exact_norms[i], exact_norms[jpvt[i]]);

        //
        //          Apply previous Householder reflectors to column K:
        //          A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**H.
        //
        // A2 := A2 - A1 F1^H
        auto A1 = slice(A, pair{i, m}, pair{0, i});
        auto A2 = slice(A, pair{i, m}, pair{i, i + 1});
        auto F1 = slice(F, pair{i, i + 1}, pair{0, i});
        gemm(Op::NoTrans, Op::ConjTrans, -one, A1, F1, one, A2);

        //
        //          Generate elementary reflector H(k).
        //
        // Transform A2 into a Householder reflector
        auto v = slice(A, pair{i, m}, i);
        larfg(forward, columnwise_storage, v, tau[i]);
        T Aii = A(i, i);
        A(i, i) = one;

        //
        //          Compute Kth column of F:
        //          Compute  F(K+1:N,K) :=
        //          tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
        //
        // F2 := tau_i A3^H A2
        auto A3 = slice(A, pair{i, m}, pair{i + 1, n});
        auto F2 = slice(F, pair{i + 1, n}, pair{i, i + 1});
        gemm(Op::ConjTrans, Op::NoTrans, tau[i], A3, A2, F2);

        //
        //          Padding F(1:K,K) with zeros.
        //
        for (idx_t j = 0; j <= i; j++) {
            F(j, i) = zero;
        }

        //
        //          Incremental updating of F:
        //              F(1:N,K) := F(1:N,K) - tau(K) * F(1:N,1:K-1) *
        //              A(RK:M,1:K-1)**H * A(RK:M,K)
        //
        // F4 := F4 - tau_i F3 A1^H A2
        //
        // auxv1 := -tau_i A1^H A2
        auto auxv1 = slice(auxv, pair{0, i}, pair{0, 1});
        gemm(Op::ConjTrans, Op::NoTrans, -tau[i], A1, A2, auxv1);
        // F4 := F4 + F3 auxv1
        auto F3 = slice(F, pair{0, n}, pair{0, i});
        auto F4 = slice(F, pair{0, n}, pair{i, i + 1});
        gemm(Op::NoTrans, Op::NoTrans, one, F3, auxv1, one, F4);

        //
        //          Update the current row of A:
        //              A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K) *
        //              F(K+1:N,1:K)**H
        //
        // A5 := A5 - A4 F5^H
        auto A4 = slice(A, pair{i, i + 1}, pair{0, i + 1});
        auto A5 = slice(A, pair{i, i + 1}, pair{i + 1, n});
        auto F5 = slice(F, pair{i + 1, n}, pair{0, i + 1});
        gemm(Op::NoTrans, Op::ConjTrans, -one, A4, F5, one, A5);

        //
        //        Update partial column norms
        //
        for (idx_t j = i + 1; j < n; j++) {
            //  // => need review: I do not think we need rzero and rone, we
            //  can use 0 and 1 directly

            if (partial_norms[j] != zero) {
                //                  NOTE: The following 4 lines follow from
                //                  the analysis in Lapack Working Note 176.
                real_t temp, temp2;

                temp = tlapack::abs(A(i, j)) / partial_norms[j];
                temp = max(zero, (one + temp) * (one - temp));
                temp2 = partial_norms[j] / exact_norms[j];
                temp2 = temp * (temp2 * temp2);
                if (temp2 <= tol3z) {
                    if (i + 1 < m) {
                        partial_norms[j] = nrm2(slice(A, pair{i + 1, m}, j));
                        exact_norms[j] = partial_norms[j];
                    }
                    else {
                        partial_norms[j] = zero;
                        exact_norms[j] = zero;
                    }
                }
                else {
                    partial_norms[j] = partial_norms[j] * sqrt(temp);
                }
            }
        }
        //
        A(i, i) = Aii;
        //
    }

    //
    //  Apply the block reflector to the rest of the matrix:
    //  A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
    //      A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**H.
    //
    auto tilA = slice(A, pair{nb, m}, pair{nb, n});
    auto V = slice(A, pair{nb, m}, pair{0, nb});
    auto tilF = slice(F, pair{nb, n}, pair{0, nb});
    gemm(Op::NoTrans, Op::ConjTrans, -one, V, tilF, one, tilA);

    //
    //  TODO: Recomputation of difficult columns.
    //

    return 0;
}

/** Computes a QR factorization of a matrix A.
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
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the min(m,n)-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *      - @c opts.work is used if whenever it has sufficient size.
 *        The sufficient size can be obtained through a workspace query.
 *
 * @ingroup computational
 */
template <class matrix_t, class vector_idx, class vector_t>
int laqp3(matrix_t& A,
          vector_idx& jpvt,
          vector_t& tau,
          const workspace_opts_t<>& opts = {})
{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kk = std::min<idx_t>(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < std::min<idx_t>(m, n));

    // quick return
    if (n <= 0) return 0;

    //  // => need review: have vector_of_norms as part of the workspace
    //  this will need to be removed
    std::vector<real_t> vector_of_norms(2 * n);

    for (idx_t j = 0; j < n; j++) {
        vector_of_norms[j] = nrm2(col(A, j));
        vector_of_norms[n + j] = vector_of_norms[j];
    }

    idx_t nb = 3;

    for (idx_t ii = 0; ii < kk; ii += nb) {
        idx_t offset = ii;
        idx_t ib = std::min<idx_t>(nb, kk - ii);

        auto Akk = slice(A, pair{offset, m}, pair{offset, n});
        auto jpvtk = slice(jpvt, pair{offset, offset + ib});
        auto tauk = slice(tau, pair{offset, offset + ib});
        auto partial_normsk = slice(vector_of_norms, pair{offset, n});
        auto exact_normsk = slice(vector_of_norms, pair{n + offset, 2 * n});

        laqps(Akk, jpvtk, tauk, partial_normsk, exact_normsk);

        // TODO: Swap the columns above Akk
        auto A0k = slice(A, pair{0, offset}, pair{offset, n});
        for (idx_t j = 0; j != ib; j++) {
            auto vect1 = tlapack::col(A0k, j);
            auto vect2 = tlapack::col(A0k, jpvtk[j]);
            tlapack::swap(vect1, vect2);
        }

        for (idx_t j = 0; j != ib; j++) {
            jpvtk[j] += offset;
        }

        // for (idx_t i = ii; i < ii + ib; ++i) {
        //     idx_t k = i - ii;
        //     idx_t rk = i;
        //     std::cout << "ii = " << ii << "; i = " << i << "; rk = " << rk
        //               << "; k = " << k << "\n";
        //     //
        //     //          Determine ith pivot column and swap if necessary
        //     //
        //     jpvtk[k] = k;
        //     for (idx_t j = k + 1; j < n; j++) {
        //         if (vector_of_normsk[j] > vector_of_normsk[jpvtk[k]])
        //             jpvtk[k] = j;
        //     }
        //     auto ak = col(Akk, k);
        //     auto bk = col(Akk, jpvtk[k]);
        //     tlapack::swap(ak, bk);
        //     auto frow1 = row(F, k);
        //     auto frow2 = row(F, jpvtk[k]);
        //     tlapack::swap(frow1, frow2);
        //     std::swap(vector_of_normsk[k], vector_of_normsk[jpvtk[k]]);
        //     std::swap(vector_of_normsk[n + k], vector_of_normsk[n +
        //     jpvtk[k]]);
        //     //
        //     //          Apply previous Householder reflectors to column K:
        //     //          A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**H.
        //     //
        //     auto A1 = slice(A, pair{rk, m}, pair{0, k});
        //     auto A2 = slice(A, pair{rk, m}, pair{k, k + 1});
        //     auto F1 = slice(F, pair{k, k + 1}, pair{0, k});
        //     gemm(Op::NoTrans, Op::ConjTrans, -one, A1, F1, one, A2);
        //     //
        //     //          Generate elementary reflector H(k).
        //     //
        //     auto v = slice(A, pair{k, m}, k);
        //     larfg(v, tau[k]);
        //     //
        //     T Aii = A(i, i);
        //     A(i, i) = one;
        //     //
        //     //          Compute Kth column of F:
        //     //          Compute  F(K+1:N,K) :=
        //     //          tau(K)*A(RK:M,K+1:N)**H*A(RK:M,K).
        //     //
        //     if (i + 1 < n) {
        //         auto A3 = slice(A, pair{i, m}, pair{i + 1, n});
        //         auto A4 = slice(A, pair{i, m}, pair{i, i + 1});
        //         auto F2 = slice(F, pair{i + 1, n}, pair{i - ii, i - ii + 1});
        //         gemm(Op::ConjTrans, Op::NoTrans, tau[i], A3, A4, F2);
        //     }
        //     //
        //     //          Padding F(1:K,K) with zeros.
        //     //
        //     for (idx_t j = 0; j < i; j++) {
        //         F(j, i - ii) = zero;
        //     }
        //     //
        //     //          Incremental updating of F:
        //     //              F(1:N,K) := F(1:N,K) - tau(K) * F(1:N,1:K-1) *
        //     //              A(RK:M,1:K-1)**H * A(RK:M,K)
        //     //
        //     if (ii < i) {
        //         auto A5 = slice(A, pair{i, m}, pair{ii, i});
        //         auto A6 = slice(A, pair{i, m}, pair{i, i + 1});
        //         auto F3 = slice(F, pair{0, n}, pair{0, i - ii});
        //         auto F4 = slice(F, pair{0, n}, pair{i - ii, i - ii + 1});
        //         auto auxv1 = slice(auxv, pair{0, i - ii}, pair{0, 1});
        //         gemm(Op::ConjTrans, Op::NoTrans, -tau[i], A5, A6, auxv1);
        //         gemm(Op::NoTrans, Op::NoTrans, one, F3, auxv1, one, F4);
        //     }
        //     //
        //     //          Update the current row of A:
        //     //              A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K) *
        //     //              F(K+1:N,1:K)**H
        //     //
        //     if (i + 1 < n) {
        //         auto A7 = slice(A, pair{i, i + 1}, pair{0, i - ii});
        //         auto A8 = slice(A, pair{i, i + 1}, pair{i + 1, n});
        //         auto F5 = slice(F, pair{i + 1, n}, pair{0, i - ii});
        //         gemm(Op::NoTrans, Op::ConjTrans, -one, A7, F5, one, A8);
        //     }
        //     //
        //     //        Update partial column norms
        //     //
        //     for (idx_t j = i + 1; j < n; j++) {
        //         //  // => need review: I do not think we need rzero and rone,
        //         we
        //         //  can use 0 and 1 directly

        //         if (vector_of_norms[j] != zero) {
        //             //                  NOTE: The following 4 lines follow
        //             from
        //             //                  the analysis in Lapack Working Note
        //             176. real_t temp, temp2; const real_t rone(1);

        //             temp = std::abs(A(i, j)) / vector_of_norms[j];
        //             temp = max(zero, (rone + temp) * (rone - temp));
        //             temp2 = vector_of_norms[j] / vector_of_norms[n + j];
        //             temp2 = temp * (temp2 * temp2);
        //             if (temp2 <= tol3z) {
        //                 if (i + 1 < m) {
        //                     vector_of_norms[j] =
        //                         nrm2(slice(A, pair{i + 1, m}, j));
        //                     vector_of_norms[n + j] = vector_of_norms[j];
        //                 }
        //                 else {
        //                     vector_of_norms[j] = 0;
        //                     vector_of_norms[n + j] = 0;
        //                 }
        //             }
        //             else {
        //                 vector_of_norms[j] =
        //                     vector_of_norms[j] * std::sqrt(temp);
        //             }
        //         }
        //     }
        //     //
        //     A(i, i) = Aii;
        //     //
        // }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAQPS_HH