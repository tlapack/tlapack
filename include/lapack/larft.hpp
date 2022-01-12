/// @file larft.hpp Forms the triangular factor T of a block reflector.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larft.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARFT_HH__
#define __LARFT_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "tblas.hpp"

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
template< 
    class direction_t, class storage_t,
    class matrixV_t, class vector_t, class matrixT_t,
    enable_if_t<(
    /* Requires: */
    (
        is_same_v< direction_t, forward_t > || 
        is_same_v< direction_t, backward_t > 
    ) && (
        is_same_v< storage_t, columnwise_storage_t > || 
        is_same_v< storage_t, rowwise_storage_t >
    )
    ), int > = 0
>
int larft(
    direction_t direction, storage_t storeMode,
    const matrixV_t& V, const vector_t& tau, matrixT_t& T)
{
    // data traits
    using scalar_t  = type_t< matrixT_t >;
    using tau_t     = type_t< vector_t >;
    using idx_t     = size_type< matrixV_t >;

    // using
    using blas::conj;
    using blas::max;
    using blas::min;
    using blas::gemm;
    using blas::gemv;
    using blas::trmv;
    using pair = std::pair<idx_t,idx_t>;

    // constants
    const scalar_t one(1);
    const scalar_t zero(0);
    const tau_t    tzero(0);
    const auto n    = (is_same_v< storage_t, columnwise_storage_t >)
                    ? nrows( V )
                    : ncols( V );
    const auto k    = (is_same_v< storage_t, columnwise_storage_t >)
                    ? ncols( V )
                    : nrows( V );

    // check arguments
    lapack_error_if( size( tau ) != k, -4 );
    lapack_error_if( nrows( T ) != k ||
                     ncols( T ) != k, -5 );

    // Quick return
    if (n == 0 || k == 0)
        return 0;

    if (is_same_v< direction_t, forward_t >) {
        T(0,0) = tau(0);
        for (idx_t i = 1; i < k; ++i) {
            auto Ti = col( T, i, pair(0,i) );
            if (tau(i) == tzero) {
                // H(i)  =  I
                for (idx_t j = 0; j <= i; ++j)
                    T(j,i) = zero;
            }
            else {
                // General case
                if (is_same_v< storage_t, columnwise_storage_t >) {
                    for (idx_t j = 0; j < i; ++j)
                        T(j,i) = -tau(i) * conj(V(i,j));
                    // T(0:i,i) := - tau(i) V(i+1:n,0:i)^H V(i+1:n,i)
                    gemv( conjTranspose,
                        -tau(i),
                        submatrix( V, pair(i+1,n), pair(0,i) ),
                        col( V, i, pair(i+1,n) ),
                        one, Ti
                    );
                }
                else {
                    for (idx_t j = 0; j < i; ++j)
                        T(j,i) = -tau(i) * V(j,i);
                    // T(0:i,i) := - tau(i) V(0:i,i:n) V(i,i+1:n)^H
                    if( is_complex<scalar_t>::value ) {
                        auto matrixTi = submatrix( T, pair(0,i), pair(i,i+1) );
                        gemm( noTranspose, conjTranspose,
                            -tau(i),
                            submatrix( V, pair(0,i), pair(i+1,n) ),
                            submatrix( V, pair(i,i+1), pair(i+1,n) ),
                            one, matrixTi
                        );
                    } else {
                        gemv( noTranspose,
                            -tau(i),
                            submatrix( V, pair(0,i), pair(i+1,n) ),
                            row( V, i, pair(i+1,n) ),
                            one, Ti
                        );
                    }
                }
                // T(0:i,i) := T(0:i,0:i) * T(0:i,i)
                trmv( upper_triangle, noTranspose, nonUnit_diagonal,
                    submatrix( T, pair(0,i), pair(0,i) ), Ti 
                );
                T(i,i) = tau(i);
            }
        }
    }
    else { // direct==Direction::Backward
        T(k-1,k-1) = tau(k-1);
        for (idx_t i = k-2; i != idx_t(-1); --i) {
            auto Ti = col( T, i, pair(i+1,k) );
            if (tau(i) == tzero) {
                for (idx_t j = i; j < k; ++j)
                    T(j,i) = zero;
            }
            else {
                if (is_same_v< storage_t, columnwise_storage_t >) {
                    for (idx_t j = i+1; j < k; ++j)
                        T(j,i) = -tau(i) * conj(V(n-k+i,j));
                    // T(i+1:k,i) := - tau(i) V(0:n-k+i,i+1:k)^H V(0:n-k+i,i)
                    gemv( conjTranspose,
                        -tau(i),
                        submatrix( V, pair(0,n-k+i), pair(i+1,k) ),
                        col( V, i, pair(0,n-k+i) ),
                        one, Ti
                    );
                }
                else {
                    for (idx_t j = i+1; j < k; ++j)
                        T(j,i) = -tau(i) * V(j,n-k+i);
                    // T(i+1:k,i) := - tau(i) V(i+1:k,0:n-k+i) V(i,0:n-k+i)^H
                    if( blas::is_complex<scalar_t>::value ) {
                        auto matrixTi = submatrix( T, pair(i+1,k), pair(i,i+1) );
                        gemm( noTranspose, conjTranspose,
                            -tau(i),
                            submatrix( V, pair(i+1,k), pair(0,n-k+i) ),
                            submatrix( V, pair(i,i+1), pair(0,n-k+i) ),
                            one, matrixTi
                        );
                    } else {
                        gemv( noTranspose,
                            -tau(i),
                            submatrix( V, pair(i+1,k), pair(0,n-k+i) ),
                            row( V, i, pair(0,n-k+i) ),
                            one, Ti
                        );
                    }
                }
                trmv( lower_triangle, noTranspose, nonUnit_diagonal,
                    submatrix( T, pair(i+1,k), pair(i+1,k) ), Ti 
                );
                T(i,i) = tau(i);
            }
        }
    }
    return 0;
}

}

#endif // __LARFT_HH__
