/// @file larft.hpp Forms the triangular factor T of a block reflector.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LEGACY_LARFT_HH
#define TLAPACK_LEGACY_LARFT_HH

#include "tlapack/lapack/larft.hpp"

namespace tlapack {

/** Forms the triangular factor T of a block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 *               If direction = Direction::Forward, H = H_1 H_2 . . . H_k and T
 * is upper triangular. If direction = Direction::Backward, H = H_k . . . H_2
 * H_1 and T is lower triangular.
 *
 *  If storev = StoreV::Columnwise, the vector which defines the elementary
 * reflector H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 *  If storev = StoreV::Rowwise, the vector which defines the elementary
 * reflector H(i) is stored in the i-th row of the array V, and
 *
 *               H  =  I - V' * T * V
 *
 *  The shape of the matrix V and the storage of the vectors which define
 *  the H(i) is best illustrated by the following example with n = 5 and
 *  k = 3. The elements equal to 1 are not stored.
 *
 *               direction=Direction::Forward & storev=StoreV::Columnwise
 * direction=Direction::Forward & storev=StoreV::Rowwise
 *               -----------------------          -----------------------
 *               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *                   ( v1  1    )                     (     1 v2 v2 v2 )
 *                   ( v1 v2  1 )                     (        1 v3 v3 )
 *                   ( v1 v2 v3 )
 *                   ( v1 v2 v3 )
 *
 *               direction=Direction::Backward & storev=StoreV::Columnwise
 * direction=Direction::Backward & storev=StoreV::Rowwise
 *               -----------------------          -----------------------
 *               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *                   (     1 v3 )
 *                   (        1 )
 *
 * @return 0 if success.
 * @return -i if the ith argument is invalid.
 *
 * @param direction Specifies the direction in which the elementary reflectors
 * are multiplied to form the block reflector.
 *
 *               Direction::Forward
 *               Direction::Backward
 *
 * @param storev Specifies how the vectors which define the elementary
 * reflectors are stored.
 *
 *               StoreV::Columnwise
 *               StoreV::Rowwise
 *
 * @param n The order of the block reflector H. n >= 0.
 * @param k The order of the triangular factor T, or the number of elementary
 * reflectors. k >= 1.
 * @param[in] V Real matrix containing the vectors defining the elementary
 * reflector H. If stored columnwise, V is n-by-k.  If stored rowwise, V is
 * k-by-n.
 * @param ldV Column length of the matrix V.  If stored columnwise, ldV >= n.
 * If stored rowwise, ldV >= k.
 * @param[in] tau Real vector of length k containing the scalar factors of the
 * elementary reflectors H.
 * @param[out] T Real matrix of size k-by-k containing the triangular factor of
 * the block reflector. If the direction of the elementary reflectors is
 * forward, T is upper triangular; if the direction of the elementary reflectors
 * is backward, T is lower triangular.
 * @param ldT Column length of the matrix T.  ldT >= k.
 *
 * @ingroup legacy_lapack
 */
template <class direction_t, class storeV_t, typename scalar_t>
int larft(direction_t direction,
          storeV_t storev,
          idx_t n,
          idx_t k,
          const scalar_t* V,
          idx_t ldV,
          const scalar_t* tau,
          scalar_t* T,
          idx_t ldT)
{
    using internal::colmajor_matrix;
    using internal::vector;

    // check arguments
    tlapack_check_false(direction != Direction::Forward &&
                        direction != Direction::Backward);
    tlapack_check_false(storev != StoreV::Columnwise &&
                        storev != StoreV::Rowwise);
    tlapack_check_false(n < 0);
    tlapack_check_false(k < 1);
    tlapack_check_false(ldV < ((storev == StoreV::Columnwise) ? n : k));
    tlapack_check_false(ldT < k);

    // Quick return
    if (n == 0 || k == 0) return 0;

    // Matrix views
    auto V_ = (storev == StoreV::Columnwise)
                  ? colmajor_matrix<scalar_t>((scalar_t*)V, n, k, ldV)
                  : colmajor_matrix<scalar_t>((scalar_t*)V, k, n, ldV);
    auto tau_ = vector((scalar_t*)tau, k);
    auto T_ = colmajor_matrix<scalar_t>(T, k, k, ldT);

    return larft(direction, storev, V_, tau_, T_);
}

}  // namespace tlapack

#endif  // TLAPACK_LEGACY_LARFT_HH
