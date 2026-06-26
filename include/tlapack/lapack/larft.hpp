/// @file larft.hpp Forms the triangular factor T of a block reflector.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LARFT_HH
#define TLAPACK_LARFT_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larft3.hpp"
#include "tlapack/lapack/larft3_recursive.hpp"
#include "tlapack/lapack/larft_legacy.hpp"
#include "tlapack/lapack/larft_recursive_legacy.hpp"

namespace tlapack {

/// @brief Variants larft.
enum class LarftVariant : char { General = 'G', Recursive = 'R' };

/// @brief Options struct for larft()
struct LarftOpts {
    LarftVariant variant = LarftVariant::General;
};

/**
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @param[in] tau Vector of length k containing the scalar factors
 *      of the elementary reflectors H.
 *
 * @param[out] T Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs, and nb for potrf_blocked.
 *      - variant:
 *          - Recursive = 'R'
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX matrixT_t>
int larft(direction_t direction,
          storage_t storeMode,
          const matrixV_t& V,
          const vector_t& tau,
          matrixT_t& T,
          const LarftOpts& opts = {})
{
    if (opts.variant == LarftVariant::Recursive) {
        return larft_recursive_legacy(direction, storeMode, V, tau, T);
    }
    else {
        return larft_legacy(direction, storeMode, V, tau, T);
    }
}

/**
 * @tparam direction_t Either Direction or any class that implements `operator
 * Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator
 * StoreV()`.
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are
 * stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise: n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:    k-by-n matrix V.
 *
 * @param[out] T Matrix of size k-by-k containing the triangular factors
 *      of the block reflector.
 *     - Direction::Forward:  T is upper triangular.
 *     - Direction::Backward: T is lower triangular.
 *
 * @param[in] opts Options.
 *      Define the behavior of checks for NaNs, and nb for potrf_blocked.
 *      - variant:
 *          - Recursive = 'R'
 *
 * @ingroup auxiliary
 */
template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_SMATRIX matrixV_t,
          TLAPACK_SMATRIX matrixT_t>
int larft(direction_t direction,
          storage_t storeMode,
          const matrixV_t& V,
          matrixT_t& T,
          const LarftOpts& opts = {})
{
    if (opts.variant == LarftVariant::Recursive) {
        return larft3_recursive(direction, storeMode, V, T);
    }
    else {
        return larft3(direction, storeMode, V, T);
    }
}

}  // namespace tlapack

#endif  // TLAPACK_LARFT_HH
