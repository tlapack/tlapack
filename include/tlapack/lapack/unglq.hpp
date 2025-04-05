/// @file unglq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zunglq.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGLQ_HH
#define TLAPACK_UNGLQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ungq.hpp"

namespace tlapack {

/**
 * Options struct for unglq
 */
struct UnglqOpts {
    size_t nb = 32;  ///< Block size
};

/**
 * Generates all or part of the unitary matrix Q from an LQ factorization
 * determined by gelqf.
 *
 * The matrix Q is defined as the first k rows of a product of k elementary
 * reflectors of order n
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H
 * \]
 * as returned by gelqf and k <= n.
 *
 * @return  0 if success
 *
 * @param[in,out] A k-by-n matrix.
 *      On entry, the i-th row must contain the vector which defines
 *      the elementary reflector H(j), for j = 1,2,...,k, as returned
 *      by gelq2 in the first k rows of its array argument A.
 *      On exit, the k by n matrix Q.
 *
 * @param[in] tau Complex vector of length min(m,n).
 *      tau(j) must contain the scalar factor of the elementary
 *      reflector H(j), as returned by gelqf.
 *
 * @param[in] opts Options.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int unglq(matrix_t& A, const vector_t& tau, const UnglqOpts& opts = {})
{
    return ungq(FORWARD, ROWWISE_STORAGE, A, tau, UngqOpts{opts.nb});
}

}  // namespace tlapack

#endif  // TLAPACK_UNGLQ_HH
