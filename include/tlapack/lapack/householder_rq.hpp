/// @file householder_rq.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HOUSEHOLDER_RQ_HH
#define TLAPACK_HOUSEHOLDER_RQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/gerq2.hpp"
#include "tlapack/lapack/gerqf.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to compute the RQ factorization.
enum class HouseholderRQVariant : char { Level2 = '2', Blocked = 'B' };

/// @brief Options struct for householder_rq()
struct HouseholderRQOpts : public GerqfOpts {
    HouseholderRQVariant variant = HouseholderRQVariant::Blocked;
};

/** Worspace query of householder_rq()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] tau min(n,m) vector.
 *
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo householder_rq_worksize(const matrix_t& A,
                                           const vector_t& tau,
                                           const HouseholderRQOpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderRQVariant::Level2)
        return gerq2_worksize<T>(A, tau);
    else
        return gerqf_worksize<T>(A, tau, opts);
}

/** @copybrief householder_rq()
 * Workspace is provided as an argument.
 * @copydetails householder_rq()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE workspace_t>
int householder_rq_work(matrix_t& A,
                        vector_t& tau,
                        workspace_t& work,
                        const HouseholderRQOpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderRQVariant::Level2)
        return gerq2_work(A, tau, work);
    else
        return gerqf_work(A, tau, work, opts);
}

/** Computes a RQ factorization of an m-by-n matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_k ... H_2 H_1,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[n-k+i+1:n] = 0; v[n-k+i-1] = 1,
 * \]
 * with conj(v[1]) through conj(v[n-k+i-1]) stored on exit in
 * A(m-k+i,0:n-k+i-1), and conj(tau) in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, if m <= n, the upper triangle of the subarray
 *      A(0:m,n-m:n) contains the m by m upper triangular matrix R;
 *      if m >= n, the elements on and above the (m-n)-th subdiagonal
 *      contain the m by n upper trapezoidal matrix R; the remaining
 *      elements, with the array tau, represent the unitary matrix
 *      Q as a product of elementary reflectors.
 *
 * @param[out] tau Real vector of length k.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
int householder_rq(matrix_t& A,
                   vector_t& tau,
                   const HouseholderRQOpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderRQVariant::Level2)
        return gerq2(A, tau);
    else
        return gerqf(A, tau, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_HOUSEHOLDER_RQ_HH