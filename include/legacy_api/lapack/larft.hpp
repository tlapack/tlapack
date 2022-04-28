/// @file larft.hpp Forms the triangular factor T of a block reflector.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LEGACY_LARFT_HH__
#define __TLAPACK_LEGACY_LARFT_HH__

#include "lapack/larft.hpp"

namespace lapack {

/** Forms the triangular factor T of a block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 *               If direct = Direction::Forward, H = H_1 H_2 . . . H_k and T is upper triangular.
 *               If direct = Direction::Backward, H = H_k . . . H_2 H_1 and T is lower triangular.
 *
 *  If storeV = StoreV::Columnwise, the vector which defines the elementary reflector
 *  H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 *  If storeV = StoreV::Rowwise, the vector which defines the elementary reflector
 *  H(i) is stored in the i-th row of the array V, and
 *
 *               H  =  I - V' * T * V
 *
 *  The shape of the matrix V and the storage of the vectors which define
 *  the H(i) is best illustrated by the following example with n = 5 and
 *  k = 3. The elements equal to 1 are not stored.
 *
 *               direct=Direction::Forward & storeV=StoreV::Columnwise          direct=Direction::Forward & storeV=StoreV::Rowwise
 *               -----------------------          -----------------------
 *               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *                   ( v1  1    )                     (     1 v2 v2 v2 )
 *                   ( v1 v2  1 )                     (        1 v3 v3 )
 *                   ( v1 v2 v3 )
 *                   ( v1 v2 v3 )
 *
 *               direct=Direction::Backward & storeV=StoreV::Columnwise          direct=Direction::Backward & storeV=StoreV::Rowwise
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
 * @param direct Specifies the direction in which the elementary reflectors are multiplied to form the block reflector.
 *
 *               Direction::Forward
 *               Direction::Backward
 *
 * @param storeV Specifies how the vectors which define the elementary reflectors are stored.
 *
 *               StoreV::Columnwise
 *               StoreV::Rowwise
 *
 * @param n The order of the block reflector H. n >= 0.
 * @param k The order of the triangular factor T, or the number of elementary reflectors. k >= 1.
 * @param[in] V Real matrix containing the vectors defining the elementary reflector H.
 * If stored columnwise, V is n-by-k.  If stored rowwise, V is k-by-n.
 * @param ldV Column length of the matrix V.  If stored columnwise, ldV >= n.
 * If stored rowwise, ldV >= k.
 * @param[in] tau Real vector of length k containing the scalar factors of the elementary reflectors H.
 * @param[out] T Real matrix of size k-by-k containing the triangular factor of the block reflector.
 * If the direction of the elementary reflectors is forward, T is upper triangular;
 * if the direction of the elementary reflectors is backward, T is lower triangular.
 * @param ldT Column length of the matrix T.  ldT >= k.
 * 
 * @ingroup auxiliary
 */
template <class direction_t, class storeV_t, typename scalar_t>
int larft(
    direction_t direction, storeV_t storeV,
    idx_t n, idx_t k,
    const scalar_t *V, idx_t ldV,
    const scalar_t *tau,
    scalar_t *T, idx_t ldT)
{
    using blas::internal::colmajor_matrix;
    using blas::internal::vector;

    // check arguments
    lapack_error_if( direction != Direction::Forward &&
                     direction != Direction::Backward, -1 );
    lapack_error_if( storeV != StoreV::Columnwise &&
                     storeV != StoreV::Rowwise, -2 );
    lapack_error_if( n < 0, -3 );
    lapack_error_if( k < 1, -4 );
    lapack_error_if( ldV < ((storeV == StoreV::Columnwise) ? n : k), -6 );
    lapack_error_if( ldT < k, -9 );

    // Quick return
    if (n == 0 || k == 0)
        return 0;

    // Matrix views
    auto _V = (storeV == StoreV::Columnwise)
            ? colmajor_matrix<scalar_t>( (scalar_t*)V, n, k, ldV )
            : colmajor_matrix<scalar_t>( (scalar_t*)V, k, n, ldV );
    auto _tau = vector( (scalar_t*)tau, k );
    auto _T = colmajor_matrix<scalar_t>( T, k, k, ldT );

    return larft( direction, storeV, _V, _tau, _T);
}

}

#endif // __TLAPACK_LEGACY_LARFT_HH__
