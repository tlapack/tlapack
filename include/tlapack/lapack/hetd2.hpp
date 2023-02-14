/// @file hetd2.hpp
/// @author Skylar Johns, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HETD2_HH
#define TLAPACK_HETD2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/symv.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of geqr2()
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
template <class matrix_t, class vector_t>
inline constexpr void geqr2_worksize(const matrix_t& A,
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
template <class matrix_t, class vector_t, class uplo_t>
int hetd2(uplo_t uplo,
          matrix_t& A,
          vector_t& tau,
          const workspace_opts_t<>& opts = {})

{
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using idx_t = size_type<matrix_t>;
    using pair = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);
    const real_t one(1);
    const real_t zero(0);

    // check arguments
    tlapack_check((idx_t)size(tau) >= n - 1);

    // check arguments
    tlapack_check(nrows(A) == ncols(A));
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    vectorOfBytes localworkdata;
    Workspace work = [&]() {
        workinfo_t workinfo;
        geqr2_worksize(A, tau, workinfo, opts);
        return alloc_workspace(localworkdata, workinfo, opts.work);
    }();

    // Options to forward
    auto&& larfOpts = workspace_opts_t<>{work};

    if (uplo == Uplo::Upper) {
        //
        // Reduce upper triangle of A
        //
        for (idx_t i = n - 2; i != -1; --i) {
            // Define x := A[0:i,i+1]
            auto x = slice(A, pair{0, i}, i + 1);
            // Generate elementary reflector H(i) = I - tau * v * v**T
            // to annihilate A(0:i,i+1)
            larfg(A(i, i + 1), x, tau[i]);
            if (tau[i] != zero) {
                // Apply H(i) from both sides to A(0:i, 0:i)
                auto C = slice(A, pair{0, i}, pair{0, i});
                A(i, i + 1) = one;
                // Compute x:= tau * A * v storing x in TAU(1:i)
                // symv(
            }
        }
    }
    else {
        for (idx_t i = 0; i < n - 1; ++i) {
            // Define v := A[i:(n-1),i]
            auto v = slice(A, pair{i + 1, n}, i);
            // Generate (i+1) reflector
            larfg(v, tau[i]);

            if (i + 1 < n) {
                // Define v:= A[i:k,i] and C:= A[i:k,i+1:n]
                auto C = slice(A, pair{i, n}, pair{i + 1, n});
                // Apply H(i) to the remainder of A
            }
        }
    }

    //     auto C = slice( A, pair{i,m}, pair{i+1,n} );

    //     // C := ( I - conj(tau_i) v v^H ) C
    //     larf( left_side, forward, v, conj(tau[i]), C, larfOpts );
    // }
    // if( n-1 < m ) {
    //     // Define v := A[n-1:m,n-1]
    //     auto v = slice( A, pair{n-1,m}, n-1 );
    //     // Generate the n-th elementary Householder reflection on x
    //     larfg( v, tau[n-1] );
    // }

    // //
    // // Reduce the lower triangle of A
    // //
    // for(idx_t i = n - 2; i != -1; --i){
    //     auto x = slice( A, pair{0,i}, i + 1 );
    //         // Generate elementary reflector H(i) = I - tau * v * v**T
    //        //to annihilate A(0:i,i+1)
    //     larfg( A(i, i+1), x, tau[i] );
    //     if(tau[i] != 0) {
    //         // Apply H(i) from both sides to A(0:i, 0:i)
    //         auto C = slice( A, pair{0,i}, pair{0,i} );
    //         A(i, i+1) = one;
    //         // Compute x:= tau * A * v storing x in TAU(1:i)
    // }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GEQR2_HH
