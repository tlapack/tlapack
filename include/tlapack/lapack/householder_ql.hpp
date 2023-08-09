/// @file householder_ql.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HOUSEHOLDER_QL_HH
#define TLAPACK_HOUSEHOLDER_QL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/geql2.hpp"
#include "tlapack/lapack/geqlf.hpp"

namespace tlapack {

enum class HouseholderQLVariant : char { Level2 = '2', Blocked = 'B' };

struct HouseholderQLOpts : public GeqlfOpts {
    HouseholderQLVariant variant = HouseholderQLVariant::Blocked;
};

/** Worspace query of householder_ql()
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
constexpr WorkInfo householder_ql_worksize(const matrix_t& A,
                                           const vector_t& tau,
                                           const HouseholderQLOpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderQLVariant::Level2)
        return geql2_worksize<T>(A, tau);
    else
        return geqlf_worksize<T>(A, tau, opts);
}

/** Computes a QL factorization of an m-by-n matrix A.
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
 *          v[m-k+i+1:m] = 0; v[m-k+i-1] = 1,
 * \]
 * with v[1] through v[m-k+i-1] stored on exit in A(0:m-k+i-1,n-k+i), and tau in
 * tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, if m >= n, the lower triangle of A(m-n:m,0:n) contains
 *      the n by n lower triangular matrix L;
 *      If m <= n, the elements on and below the (n-m)-th
 *      superdiagonal contain the m by n lower trapezoidal matrix L
 *      the remaining elements, with the array tau, represent the
 *      unitary matrix Q as a product of elementary reflectors.
 *
 * @param[out] tau Real vector of length k.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr int householder_ql(matrix_t& A,
                             vector_t& tau,
                             const HouseholderQLOpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderQLVariant::Level2)
        return geql2(A, tau);
    else
        return geqlf(A, tau, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_HOUSEHOLDER_QL_HH