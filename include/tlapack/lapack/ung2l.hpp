/// @file ung2l.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zung2l.f
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNG2L_HH
#define TLAPACK_UNG2L_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

/** Worspace query of ung2l()
 *
 *
 * @param[in] A m-by-n matrix.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo ung2l_worksize(const matrix_t& A, const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    if (n > 1) {
        auto&& C = cols(A, range{1, n});
        return larf_worksize<T>(LEFT_SIDE, BACKWARD, COLUMNWISE_STORAGE,
                                col(A, 0), tau[0], C);
    }
    return WorkInfo(0);
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last n columns of a product of k elementary
 *        reflectors of order m
 * \[
 *     Q  =  H_k ... H_2 H_1
 * \]
 *        The reflectors are stored in the matrix A as returned by geqlf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (n+k-i)-th column must contains the vector which defines
 the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqlf.
 *      On exit, the m-by-n matrix $Q$.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int ung2l_work(matrix_t& A, const vector_t& tau, work_t& work)
{
    return ungq_level2_work(BACKWARD, COLUMNWISE_STORAGE, A, tau, work);
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last n columns of a product of k elementary
 *        reflectors of order m
 * \[
 *     Q  =  H_k ... H_2 H_1
 * \]
 *        The reflectors are stored in the matrix A as returned by geqlf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (n+k-i)-th column must contains the vector which defines
 the
 *      elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by geqlf.
 *      On exit, the m-by-n matrix $Q$.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @return 0 if success
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int ung2l(matrix_t& A, const vector_t& tau)
{
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t n = ncols(A);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = ung2l_worksize<T>(A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return ung2l_work(A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_UNG2L_HH
