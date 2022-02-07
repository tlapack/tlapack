/// @file larfb.hpp Applies a Householder block reflector to a matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larfb.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARFB_HH__
#define __LARFB_HH__

#include "lapack/utils.hpp"
#include "lapack/types.hpp"
#include "lapack/lacpy.hpp"
#include "tblas.hpp"

namespace lapack {

/** Applies a block reflector $H$ or its conjugate transpose $H^H$ to a
 * m-by-n matrix C, from either the left or the right.
 *
 * @param[in] side
 *     - lapack::Side::Left:  apply $H$ or $H^H$ from the Left
 *     - lapack::Side::Right: apply $H$ or $H^H$ from the Right
 *
 * @param[in] trans
 *     - lapack::Op::NoTrans:   apply $H  $ (No transpose)
 *     - lapack::Op::Trans:     apply $H^T$ (Transpose, only allowed if the type of H is Real)
 *     - lapack::Op::ConjTrans: apply $H^H$ (Conjugate transpose)
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary
 *     reflectors
 *     - lapack::Direction::Forward:  $H = H(1) H(2) \dots H(k)$
 *     - lapack::Direction::Backward: $H = H(k) \dots H(2) H(1)$
 *
 * @param[in] storev
 *     Indicates how the vectors which define the elementary
 *     reflectors are stored:
 *     - lapack::StoreV::Columnwise
 *     - lapack::StoreV::Rowwise
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
 * @param[in] W
 *     Workspace array with length
 *          k*n if side == Side::Left.
 *          k*m if side == Side::Right.
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
 * @ingroup auxiliary
 */
template<
    class matrixV_t, class matrixT_t, class matrixC_t, class matrixW_t,
    class side_t, class trans_t, class direction_t, class storage_t,
    enable_if_t<(
    /* Requires: */
    (
        is_same_v< side_t, left_side_t > || 
        is_same_v< side_t, right_side_t > 
    ) && (
        is_same_v< trans_t, noTranspose_t > || 
        is_same_v< trans_t, conjTranspose_t > ||
        is_same_v< trans_t, transpose_t >
    ) && (
        is_same_v< direction_t, forward_t > || 
        is_same_v< direction_t, backward_t > 
    ) && (
        is_same_v< storage_t, columnwise_storage_t > || 
        is_same_v< storage_t, rowwise_storage_t >
    )
    ), int > = 0
>
int larfb(
    side_t side, trans_t trans,
    direction_t direction, storage_t storeMode,
    const matrixV_t& V, const matrixT_t& T,
    matrixC_t& C, matrixW_t& W )
{
    using idx_t = size_type< matrixV_t >;
    using pair  = std::pair<idx_t,idx_t>;
    using blas::trmm;
    using blas::gemm;

    // constants
    const type_t< matrixW_t > one( 1 );

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = nrows(T);

    // check arguments
    if( is_complex< type_t< matrixV_t > >::value )
        lapack_error_if( (is_same_v< trans_t, transpose_t >), -2 );

    // Quick return
    if (m <= 0 || n <= 0) return 0;

    if( is_same_v< storage_t, columnwise_storage_t > ){
        if( is_same_v< direction_t, forward_t > ){
            if( is_same_v< side_t, left_side_t > ){
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,k} );
                const auto V2 = rows( V, pair{k,m} );
                auto C1 = rows( C, pair{0,k} );
                auto C2 = rows( C, pair{k,m} );

                // W := C1
                lacpy( general_matrix, C1, W );
                // W := V1^H W
                trmm(
                    side, Uplo::Lower,
                    Op::ConjTrans, Diag::Unit,
                    one, V1, W );
                if( m > k )
                    // W := W + V2^H C2
                    gemm(
                        Op::ConjTrans, Op::NoTrans,
                        one, V2, C2, one, W );
                // W := op(T) W
                trmm(
                    side, Uplo::Upper,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( m > k )
                    // C2 := C2 - V2 W
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        -one, V2, W, one, C2 );
                // W := - V1 W
                trmm(
                    side, Uplo::Lower,
                    Op::NoTrans, Diag::Unit,
                    -one, V1, W );

                // C1 := C1 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C1(i,j) += W(i,j);
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,k} );
                const auto V2 = rows( V, pair{k,n} );
                auto C1 = cols( C, pair{0,k} );
                auto C2 = cols( C, pair{k,n} );

                // W := C1
                lacpy( general_matrix, C1, W );
                // W := W V1
                trmm(
                    side, Uplo::Lower,
                    Op::NoTrans, Diag::Unit,
                    one, V1, W );
                if( n > k )
                    // W := W + C2 V2
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        one, C2, V2, one, W );
                // W := W op(T)
                trmm(
                    Side::Right, Uplo::Upper,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( n > k )
                    // C2 := C2 - W V2^H
                    gemm(
                        Op::NoTrans, Op::ConjTrans,
                        -one, W, V2, one, C2 );
                // W := - W V1^H
                trmm(
                    side, Uplo::Lower,
                    Op::ConjTrans, Diag::Unit,
                    -one, V1, W );
                
                // C1 := C1 + W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C1(i,j) += W(i,j);
            }
        }
        else { // direct == Direction::Backward
            if( is_same_v< side_t, left_side_t > ){
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,m-k} );
                const auto V2 = rows( V, pair{m-k,m} );
                auto C1 = rows( C, pair{0,m-k} );
                auto C2 = rows( C, pair{m-k,m} );

                // W := C2
                lacpy( general_matrix, C2, W );
                // W := V2^H W
                trmm(
                    side, Uplo::Upper,
                    Op::ConjTrans, Diag::Unit,
                    one, V2, W );
                if( m > k )
                    // W := W + V1^H C1
                    gemm(
                        Op::ConjTrans, Op::NoTrans,
                        one, V1, C1, one, W );
                // W := op(T) W
                trmm(
                    side, Uplo::Lower,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( m > k )
                    // C1 := C1 - V1 W
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        -one, V1, W, one, C1 );
                // W := - V2 W
                trmm(
                    side, Uplo::Upper,
                    Op::NoTrans, Diag::Unit,
                    -one, V2, W );

                // C2 := C2 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C2(i,j) += W(i,j);
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,n-k} );
                const auto V2 = rows( V, pair{n-k,n} );
                auto C1 = cols( C, pair{0,n-k} );
                auto C2 = cols( C, pair{n-k,n} );

                // W := C2
                lacpy( general_matrix, C2, W );
                // W := W V2
                trmm(
                    side, Uplo::Upper,
                    Op::NoTrans, Diag::Unit,
                    one, V2, W );
                if( n > k )
                    // W := W + C1 V1
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        one, C1, V1, one, W );
                // W := W op(T)
                trmm(
                    side, Uplo::Lower,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( n > k )
                    // C1 := C1 - W V1^H
                    gemm(
                        Op::NoTrans, Op::ConjTrans,
                        -one, W, V1, one, C1 );
                // W := - W V2^H
                trmm(
                    side, Uplo::Upper,
                    Op::ConjTrans, Diag::Unit,
                    -one, V2, W );
                
                // C2 := C2 + W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C2(i,j) += W(i,j);
            }
        }
    }
    else { // storeV == StoreV::Rowwise
        if( is_same_v< direction_t, forward_t > ){
            if( is_same_v< side_t, left_side_t > ){
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,k} );
                const auto V2 = cols( V, pair{k,m} );
                auto C1 = rows( C, pair{0,k} );
                auto C2 = rows( C, pair{k,m} );

                // W := C1
                lacpy( general_matrix, C1, W );
                // W := V1 W
                trmm(
                    side, Uplo::Upper,
                    Op::NoTrans, Diag::Unit,
                    one, V1, W );
                if( m > k )
                    // W := W + V2 C2
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        one, V2, C2, one, W );
                // W := op(T) W
                trmm(
                    side, Uplo::Upper,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( m > k )
                    // C2 := C2 - V2^H W
                    gemm(
                        Op::ConjTrans, Op::NoTrans,
                        -one, V2, W, one, C2 );
                // W := - V1^H W
                trmm(
                    side, Uplo::Upper,
                    Op::ConjTrans, Diag::Unit,
                    -one, V1, W );

                // C1 := C1 - W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C1(i,j) += W(i,j);
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,k} );
                const auto V2 = cols( V, pair{k,n} );
                auto C1 = cols( C, pair{0,k} );
                auto C2 = cols( C, pair{k,n} );

                // W := C1
                lacpy( general_matrix, C1, W );
                // W := W V1^H
                trmm(
                    side, Uplo::Upper,
                    Op::ConjTrans, Diag::Unit,
                    one, V1, W );
                if( n > k )
                    // W := W + C2 V2^H
                    gemm(
                        Op::NoTrans, Op::ConjTrans,
                        one, C2, V2, one, W );
                // W := W op(T)
                trmm(
                    side, Uplo::Upper,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( n > k )
                    // C2 := C2 - W V2
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        -one, W, V2, one, C2 );
                // W := - W V1
                trmm(
                    side, Uplo::Upper,
                    Op::NoTrans, Diag::Unit,
                    -one, V1, W );
                
                // C1 := C1 + W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C1(i,j) += W(i,j);
            }
        }
        else { // direct == Direction::Backward
            if( is_same_v< side_t, left_side_t > ){
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,m-k} );
                const auto V2 = cols( V, pair{m-k,m} );
                auto C1 = rows( C, pair{0,m-k} );
                auto C2 = rows( C, pair{m-k,m} );

                // W := C2
                lacpy( general_matrix, C2, W );
                // W := V2 W
                trmm(
                    side, Uplo::Lower,
                    Op::NoTrans, Diag::Unit,
                    one, V2, W );
                if( m > k )
                    // W := W + V1 C1
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        one, V1, C1, one, W );
                // W := op(T) W
                trmm(
                    side, Uplo::Lower,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( m > k )
                    // C1 := C1 - V1^H W
                    gemm(
                        Op::ConjTrans, Op::NoTrans,
                        -one, V1, W, one, C1 );
                // W := - V2^H W
                trmm(
                    side, Uplo::Lower,
                    Op::ConjTrans, Diag::Unit,
                    -one, V2, W );

                // C2 := C2 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C2(i,j) += W(i,j);
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,n-k} );
                const auto V2 = cols( V, pair{n-k,n} );
                auto C1 = cols( C, pair{0,n-k} );
                auto C2 = cols( C, pair{n-k,n} );

                // W := C2
                lacpy( general_matrix, C2, W );
                // W := W V2^H
                trmm(
                    side, Uplo::Lower,
                    Op::ConjTrans, Diag::Unit,
                    one, V2, W );
                if( n > k )
                    // W := W + C1 V1^H
                    gemm(
                        Op::NoTrans, Op::ConjTrans,
                        one, C1, V1, one, W );
                // W := W op(T)
                trmm(
                    side, Uplo::Lower,
                    trans, Diag::NonUnit,
                    one, T, W );
                if( n > k )
                    // C1 := C1 - W V1
                    gemm(
                        Op::NoTrans, Op::NoTrans,
                        -one, W, V1, one, C1 );
                // W := - W V2
                trmm(
                    side, Uplo::Lower,
                    Op::NoTrans, Diag::Unit,
                    -one, V2, W );
                
                // C2 := C2 + W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C2(i,j) += W(i,j);
            }
        }
    }

    return 0;
}

}

#endif // __LARFB_HH__
