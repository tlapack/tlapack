/// @file geqpf.hpp
/// @author Racheal Asamoah, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GEQPF_HH
#define TLAPACK_GEQPF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "tlapack/blas/swap.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

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
inline constexpr void geqpf_worksize(const matrix_t& A,
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
        larf_worksize(left_side, forward, columnwise_storage, col(A, 0), tau[0],
                      C, workinfo, opts);
    }
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
int geqpf(matrix_t& A,
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
    const idx_t k = std::min<idx_t>(m, n);

    //  // => need review: LAPACK GEQPF takes       tol3z =
    //  sqrt(dlamch('Epsilon'))
    //  // => so maybe tol3z = sqrt( 2 * eps ); ??????
    const real_t eps = ulp<real_t>();
    const real_t tol3z = sqrt(eps);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < std::min<idx_t>(m, n));

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        geqpf_worksize(A, jpvt, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    //  // => need review: have vector_of_norms as part of the workspace
    //  this will need to be removed
    std::vector<real_t> vector_of_norms(2 * n);

    for (idx_t j = 0; j < n; j++) {
        vector_of_norms[j] = nrm2(slice(A, pair{0, m}, j));
        vector_of_norms[n + j] = vector_of_norms[j];
    }

    for (idx_t i = 0; i < k; ++i) {
        jpvt[i] = i;
        for (idx_t j = i + 1; j < n; j++) {
            if (vector_of_norms[j] > vector_of_norms[jpvt[i]]) jpvt[i] = j;
        }
        auto ai = col(A, i);
        auto bi = col(A, jpvt[i]);
        tlapack::swap(ai, bi);
        std::swap(vector_of_norms[i], vector_of_norms[jpvt[i]]);
        std::swap(vector_of_norms[n + i], vector_of_norms[n + jpvt[i]]);

        // Define v := A[i:m,i]
        auto v = slice(A, pair{i, m}, i);

        // Generate the (i+1)-th elementary Householder reflection on x
        larfg(forward, columnwise_storage, v, tau[i]);

        if (i + 1 < n) {
            // Define v := A[i:m,i] and C := A[i:m,i+1:n], and w := work[i:n-1]
            auto C = slice(A, pair{i, m}, pair{i + 1, n});

            // C := ( I - conj(tau_i) v v^H ) C
            larf(left_side, forward, columnwise_storage, v, conj(tau[i]), C, larfOpts);
        }

        //      Update partial column norms
        for (idx_t j = i + 1; j < n; j++) {
            //  // => need review: I do not think we need rzero and rone, we can
            //  use 0 and 1 directly

            const real_t rzero(0);
            if (vector_of_norms[j] != rzero) {
                //              NOTE: The following 4 lines follow from the
                //              analysis in Lapack Working Note 176.
                real_t temp, temp2;
                const real_t rone(1);

                temp = std::abs(A(i, j)) / vector_of_norms[j];
                temp = max(rzero, (rone + temp) * (rone - temp));
                temp2 = vector_of_norms[j] / vector_of_norms[n + j];
                temp2 = temp * (temp2 * temp2);
                if (temp2 <= tol3z) {
                    if (i + 1 < m) {
                        vector_of_norms[j] = nrm2(slice(A, pair{i + 1, m}, j));
                        vector_of_norms[n + j] = vector_of_norms[j];
                    }
                    else {
                        vector_of_norms[j] = 0;
                        vector_of_norms[n + j] = 0;
                    }
                }
                else {
                    vector_of_norms[j] = vector_of_norms[j] * std::sqrt(temp);
                }
            }
        }
    }
    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEQPF_HH