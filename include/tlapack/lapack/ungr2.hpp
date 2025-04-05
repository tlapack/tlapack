/// @file ungr2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zungr2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGR2_HH
#define TLAPACK_UNGR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/ungq_level2.hpp"

namespace tlapack {

/** Worspace query of ungr2()
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
constexpr WorkInfo ungr2_worksize(const matrix_t& A, const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);

    if (m > 1) {
        auto&& C = rows(A, range{1, m});
        return larf_worksize<T>(RIGHT_SIDE, BACKWARD, ROWWISE_STORAGE,
                                row(A, 0), tau[0], C);
    }
    return WorkInfo(0);
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last m rows of a product of k elementary
 *        reflectors of order n
 * \[
 *     Q  =  H_1^H H_2^H ... H_k^H
 * \]
 *        The reflectors are stored in the matrix A as returned by gerqf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (m-k+i)-th row must contain the vector which
 *      defines the elementary reflector H(i), for i = 1,2,...,k, as
 *      returned by GERQF in the last k rows of its matrix argument A.
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
int ungr2_work(matrix_t& A, const vector_t& tau, work_t& work)
{
    return ungq_level2_work(BACKWARD, ROWWISE_STORAGE, A, tau, work);
}

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last m rows of a product of k elementary
 *        reflectors of order n
 * \[
 *     Q  =  H_1^H H_2^H ... H_k^H
 * \]
 *        The reflectors are stored in the matrix A as returned by gerqf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (m-k+i)-th row must contain the vector which
 *      defines the elementary reflector H(i), for i = 1,2,...,k, as
 *      returned by GERQF in the last k rows of its matrix argument A.
 *      On exit, the m-by-n matrix $Q$.

 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @return 0 if success
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int ungr2(matrix_t& A, const vector_t& tau)
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
    WorkInfo workinfo = ungr2_worksize<T>(A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return ungr2_work(A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_UNGR2_HH
