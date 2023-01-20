/// @file larfb.hpp Applies a Householder block reflector to a matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see https://github.com/langou/latl/blob/master/include/larfb.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LARFB_HH
#define TLAPACK_LEGACY_LARFB_HH

#include "tlapack/lapack/larfb.hpp"

namespace tlapack {

/** Applies a block reflector $H$ or its conjugate transpose $H^H$ to a
 * m-by-n matrix C, from either the left or the right.
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ or $H^H$ from the Left
 *     - Side::Right: apply $H$ or $H^H$ from the Right
 *
 * @param[in] trans
 *     - Op::NoTrans:   apply $H  $ (No transpose)
 *     - Op::Trans:     apply $H^T$ (Transpose, only allowed if the type of H is
 * Real)
 *     - Op::ConjTrans: apply $H^H$ (Conjugate transpose)
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary
 *     reflectors
 *     - Direction::Forward:  $H = H(1) H(2) \dots H(k)$
 *     - Direction::Backward: $H = H(k) \dots H(2) H(1)$
 *
 * @param[in] storev
 *     Indicates how the vectors which define the elementary
 *     reflectors are stored:
 *     - StoreV::Columnwise
 *     - StoreV::Rowwise
 *
 * @param[in] m
 *     The number of rows of the matrix C.
 *
 * @param[in] n
 *     The number of columns of the matrix C.
 *
 * @param[in] k
 *     The order of the matrix T (= the number of elementary
 *     reflectors whose product defines the block reflector).
 *     - If side = Left,  m >= k >= 0;
 *     - if side = Right, n >= k >= 0.
 *
 * @param[in] V
 *     - If storev = Columnwise:
 *       - if side = Left,  the m-by-k matrix V, stored in an ldv-by-k array;
 *       - if side = Right, the n-by-k matrix V, stored in an ldv-by-k array.
 *     - If storev = Rowwise:
 *       - if side = Left,  the k-by-m matrix V, stored in an ldv-by-m array;
 *       - if side = Right, the k-by-n matrix V, stored in an ldv-by-n array.
 *     - See Further Details.
 *
 * @param[in] ldv
 *     The leading dimension of the array V.
 *     - If storev = Columnwise and side = Left,  ldv >= max(1,m);
 *     - if storev = Columnwise and side = Right, ldv >= max(1,n);
 *     - if storev = Rowwise, ldv >= k.
 *
 * @param[in] T
 *     The k-by-k matrix T, stored in an ldt-by-k array.
 *     The triangular k-by-k matrix T in the representation of the
 *     block reflector.
 *
 * @param[in] ldt
 *     The leading dimension of the array T. ldt >= k.
 *
 * @param[in,out] C
 *     The m-by-n matrix C, stored in an ldc-by-n array.
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by
 *     $H C$ or $H^H C$ or $C H$ or $C H^H$.
 *
 * @param[in] ldc
 *     The leading dimension of the array C. ldc >= max(1,m).
 *
 * @par Further Details
 *
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored. The rest of the
 * array is not used.
 *
 *     direction = Forward and          direction = Forward and
 *     storev = Columnwise:             storev = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
 *         ( v1 v2 v3 )
 *         ( v1 v2 v3 )
 *
 *     direction = Backward and         direction = Backward and
 *     storev = Columnwise:             storev = Rowwise:
 *
 *     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *         (     1 v3 )
 *         (        1 )
 *
 * @ingroup legacy_lapack
 */

template <class side_t,
          class trans_t,
          class direction_t,
          class storeV_t,
          typename TV,
          typename TC>
int larfb(side_t side,
          trans_t trans,
          direction_t direction,
          storeV_t storev,
          idx_t m,
          idx_t n,
          idx_t k,
          TV const* V,
          idx_t ldv,
          TV const* T,
          idx_t ldt,
          TC* C,
          idx_t ldc)
{
    using internal::colmajor_matrix;

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::ConjTrans &&
                        ((trans != Op::Trans) || is_complex<TV>::value));
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);
    tlapack_check_false(storev != StoreV::Columnwise &&
                        storev != StoreV::Rowwise);

    // Quick return
    if (m <= 0 || n <= 0) return 0;

    // Views
    const auto V_ =
        (storev == StoreV::Columnwise)
            ? colmajor_matrix<TV>((TV*)V, (side == Side::Left) ? m : n, k, ldv)
            : colmajor_matrix<TV>((TV*)V, k, (side == Side::Left) ? m : n, ldv);
    const auto T_ = colmajor_matrix<TV>((TV*)T, k, k, ldt);
    auto C_ = colmajor_matrix<TC>(C, m, n, ldc);

    return larfb(side, trans, direction, storev, V_, T_, C_);
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LARFB_HH
