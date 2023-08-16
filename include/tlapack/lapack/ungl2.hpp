/// @file ungl2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zungl2.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGL2_HH
#define TLAPACK_UNGL2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

/** Worspace query of ungl2()
 *
 * @param[in] Q k-by-n matrix.
 *
 * @param[in] tauw Complex vector of length min(m,n).
 *      tauw(j) must contain the scalar factor of the elementary
 *      reflector H(j), as returned by gelq2.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo ungl2_worksize(const matrix_t& Q, const vector_t& tauw)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t k = nrows(Q);

    if (k > 1) {
        auto C = rows(Q, range{1, k});
        return larf_worksize<T>(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, row(Q, 0),
                                tauw[0], C);
    }
    return WorkInfo(0);
}

/**
 * Generates all or part of the unitary matrix Q from an LQ factorization
 * determined by gelq2 (unblocked algorithm).
 *
 * The matrix Q is defined as the first k rows of a product of k elementary
 * reflectors of order n
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H
 * \]
 * as returned by gelq2 and k <= n.
 *
 * @return  0 if success
 *
 * @param[in,out] Q k-by-n matrix.
 *      On entry, the i-th row must contain the vector which defines
 *      the elementary reflector H(j), for j = 1,2,...,k, as returned
 *      by gelq2 in the first k rows of its array argument A.
 *      On exit, the k by n matrix Q.
 *
 * @param[in] tauw Complex vector of length min(m,n).
 *      tauw(j) must contain the scalar factor of the elementary
 *      reflector H(j), as returned by gelq2.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int ungl2_work(matrix_t& Q, const vector_t& tauw, work_t& work)
{
    return ungq_level2_work(FORWARD, ROWWISE_STORAGE, Q, tauw, work);
}

/**
 * Generates all or part of the unitary matrix Q from an LQ factorization
 * determined by gelq2 (unblocked algorithm).
 *
 * The matrix Q is defined as the first k rows of a product of k elementary
 * reflectors of order n
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H
 * \]
 * as returned by gelq2 and k <= n.
 *
 * @return  0 if success
 *
 * @param[in,out] Q k-by-n matrix.
 *      On entry, the i-th row must contain the vector which defines
 *      the elementary reflector H(j), for j = 1,2,...,k, as returned
 *      by gelq2 in the first k rows of its array argument A.
 *      On exit, the k by n matrix Q.
 *
 * @param[in] tauw Complex vector of length min(m,n).
 *      tauw(j) must contain the scalar factor of the elementary
 *      reflector H(j), as returned by gelq2.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int ungl2(matrix_t& Q, const vector_t& tauw)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t n = ncols(Q);
    const idx_t m =
        size(tauw);  // maximum number of Householder reflectors to use

    // check arguments
    tlapack_check_false((idx_t)size(tauw) < min(m, n));

    // Allocates workspace
    WorkInfo workinfo = ungl2_worksize<T>(Q, tauw);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return ungl2_work(Q, tauw, work);
}
}  // namespace tlapack
#endif  // TLAPACK_UNGL2_HH
