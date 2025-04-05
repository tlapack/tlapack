/// @file ungrq.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungrq.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGRQ_HH
#define TLAPACK_UNGRQ_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ungq.hpp"

namespace tlapack {

/**
 * Options struct for ungrq
 */
struct UngrqOpts {
    size_t nb = 32;  ///< Block size
};

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
 * @param[in] opts Options.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int ungrq(matrix_t& A, const vector_t& tau, const UngrqOpts& opts = {})
{
    return ungq(BACKWARD, ROWWISE_STORAGE, A, tau, UngqOpts{opts.nb});
}

}  // namespace tlapack

#endif  // TLAPACK_UNGRQ_HH
