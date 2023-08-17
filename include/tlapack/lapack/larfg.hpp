/// @file larfg.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larfg.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFG_HH
#define TLAPACK_LARFG_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/nrm2.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/lapy2.hpp"
#include "tlapack/lapack/lapy3.hpp"
#include "tlapack/lapack/rscl.hpp"

namespace tlapack {

/** Generates a elementary Householder reflection.
 *
 * larfg generates a elementary Householder reflection H of order n, such that
 *
 *        H' * ( alpha ) = ( beta ),   H' * H = I.
 *             (   x   )   (   0  )
 *
 * if storeMode == StoreV::Columnwise, or
 *
 *        ( alpha x ) * H = ( beta 0 ),   H' * H = I.
 *
 * if storeMode == StoreV::Rowwise, where alpha and beta are scalars, with beta
 * real, and x is an (n-1)-element vector. H is represented in the form
 *
 *        H = I - tau * ( 1 ) * ( 1 y' )
 *                      ( y )
 *
 * if storeMode == StoreV::Columnwise, or
 *
 *        H = I - tau * ( 1  ) * ( 1 y )
 *                      ( y' )
 *
 * if storeMode == StoreV::Rowwise, where tau is a scalar and
 * y is a (n-1)-element vector. Note that H is symmetric but not hermitian.
 *
 * If the elements of x are all zero and alpha is real, then tau = 0
 * and H is taken to be the identity matrix.
 *
 * Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in,out] alpha
 *      On entry, the value alpha.
 *      On exit, it is overwritten with the value beta.
 *
 * @param[in,out] x Vector of length n-1.
 *      On entry, the vector x.
 *      On exit, it is overwritten with the vector y.
 *
 * @param[out] tau The value tau.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_STOREV storage_t, TLAPACK_VECTOR vector_t>
void larfg(storage_t storeMode,
           type_t<vector_t>& alpha,
           vector_t& x,
           type_t<vector_t>& tau)
{
    // data traits
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;

    // using
    using real_t = real_type<T>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const real_t safemin = safe_min<real_t>() / uroundoff<real_t>();
    const real_t rsafemin = one / safemin;

    // check arguments
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);

    tau = zero;
    real_t xnorm = nrm2(x);

    if (xnorm > zero || (imag(alpha) != zero)) {
        // First estimate of beta
        real_t temp = (is_real<T>) ? lapy2(real(alpha), xnorm)
                                   : lapy3(real(alpha), imag(alpha), xnorm);
        real_t beta = (real(alpha) < zero) ? temp : -temp;

        // Scale if needed
        idx_t knt = 0;
        if (abs(beta) < safemin) {
            while ((abs(beta) < safemin) && (knt < 20)) {
                knt++;
                scal(rsafemin, x);
                beta *= rsafemin;
                alpha *= rsafemin;
            }
            xnorm = nrm2(x);
            temp = (is_real<T>) ? lapy2(real(alpha), xnorm)
                                : lapy3(real(alpha), imag(alpha), xnorm);
            beta = (real(alpha) < zero) ? temp : -temp;
        }

        // compute tau and y
        tau = (beta - alpha) / beta;
        rscl(alpha - beta, x);
        if (storeMode == StoreV::Rowwise) tau = conj(tau);

        // Scale if needed
        for (idx_t j = 0; j < knt; ++j)
            beta *= safemin;

        // Store beta in alpha
        alpha = beta;
    }
}

/**
 * @brief Generates a elementary Householder reflection.
 *
 * larfg generates a elementary Householder reflection H of order n, such that
 *
 *        H' * ( alpha ) = ( beta ),   H' * H = I.
 *             (   x   )   (   0  )
 *
 * if storeMode == StoreV::Columnwise, or
 *
 *        ( alpha x ) * H = ( beta 0 ),   H' * H = I.
 *
 * if storeMode == StoreV::Rowwise, where alpha and beta are scalars, with beta
 * real, and x is an (n-1)-element vector. H is represented in the form
 *
 *        H = I - tau * ( 1 ) * ( 1 y' )
 *                      ( y )
 *
 * if storeMode == StoreV::Columnwise, or
 *
 *        H = I - tau * ( 1  ) * ( 1 y )
 *                      ( y' )
 *
 * if storeMode == StoreV::Rowwise, where tau is a scalar and
 * y is a (n-1)-element vector. Note that H is symmetric but not hermitian.
 *
 * If the elements of x are all zero and alpha is real, then tau = 0
 * and H is taken to be the identity matrix.
 *
 * Otherwise  1 <= real(tau) <= 2 and abs(tau-1) <= 1.
 *
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     v = [ alpha x ] if direction == Direction::Forward and
 *     v = [ x alpha ] if direction == Direction::Backward.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in,out] v Vector of length n.
 *      On entry,
 *          v = [ alpha x ] if direction == Direction::Forward and
 *          v = [ x alpha ] if direction == Direction::Backward.
 *      On exit,
 *          v = [ 1 y ] if direction == Direction::Forward and
 *          v = [ y 1 ] if direction == Direction::Backward.
 *
 * @param[out] tau The value tau.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<std::is_convertible_v<direction_t, Direction>, int> = 0>
void larfg(direction_t direction,
           storage_t storeMode,
           vector_t& v,
           type_t<vector_t>& tau)
{
    using idx_t = size_type<vector_t>;
    using range = pair<idx_t, idx_t>;

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);

    const idx_t alpha_idx = (direction == Direction::Forward) ? 0 : size(v) - 1;

    auto x =
        slice(v, (direction == Direction::Forward) ? range(1, size(v))
                                                   : range(0, size(v) - 1));
    type_t<vector_t> alpha = v[alpha_idx];
    larfg(storeMode, alpha, x, tau);
    v[alpha_idx] = alpha;
}

}  // namespace tlapack

#endif  // TLAPACK_LARFG_HH
