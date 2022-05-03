/// @file larfb.hpp Applies a Householder block reflector to a matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larfb.h
//
// Copyright (c) 2013-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_LARFB_HH__
#define __TLAPACK_LARFB_HH__

#include "base/utils.hpp"
#include "base/types.hpp"
#include "lapack/lacpy.hpp"
#include "tblas.hpp"

namespace tlapack {

/** Applies a block reflector $H$ or its conjugate transpose $H^H$ to a
 * m-by-n matrix C, from either the left or the right.
 * 
 * @tparam side_t Either Side or any class that implements `operator Side()`.
 * @tparam trans_t Either Op or any class that implements `operator Op()`.
 * @tparam direction_t Either Direction or any class that implements `operator Direction()`.
 * @tparam storage_t Either StoreV or any class that implements `operator StoreV()`.
 *
 * @param[in] side
 *     - Side::Left:  apply $H$ or $H^H$ from the Left.
 *     - Side::Right: apply $H$ or $H^H$ from the Right.
 *
 * @param[in] trans
 *     - Op::NoTrans:   apply $H  $ (No transpose).
 *     - Op::Trans:     apply $H^T$ (Transpose, only allowed if the type of H is Real).
 *     - Op::ConjTrans: apply $H^H$ (Conjugate transpose).
 *
 * @param[in] direction
 *     Indicates how H is formed from a product of elementary reflectors.
 *     - Direction::Forward:  $H = H(1) H(2) ... H(k)$.
 *     - Direction::Backward: $H = H(k) ... H(2) H(1)$.
 *
 * @param[in] storeMode
 *     Indicates how the vectors which define the elementary reflectors are stored:
 *     - StoreV::Columnwise.
 *     - StoreV::Rowwise.
 *     See Further Details.
 *
 * @param[in] V
 *     - If storeMode = StoreV::Columnwise:
 *       - if side = Side::Left,  the m-by-k matrix V;
 *       - if side = Side::Right, the n-by-k matrix V.
 *     - If storeMode = StoreV::Rowwise:
 *       - if side = Side::Left,  the k-by-m matrix V;
 *       - if side = Side::Right, the k-by-n matrix V.
 *
 * @param[in] T
 *     The k-by-k matrix T.
 *     The triangular k-by-k matrix T in the representation of the block reflector.
 *
 * @param[in,out] C
 *     On entry, the m-by-n matrix C.
 *     On exit, C is overwritten by $H C$ or $H^H C$ or $C H$ or $C H^H$.
 *
 * @param W Workspace matrix with length
 *     - k*n if side = Side::Left.
 *     - k*m if side = Side::Right.
 *
 * @par Further Details
 *
 * The shape of the matrix V and the storage of the vectors which define
 * the H(i) is best illustrated by the following example with n = 5 and
 * k = 3. The elements equal to 1 are not stored. The rest of the
 * array is not used.
 *
 *     direction = Forward and          direction = Forward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
 *
 *     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *         ( v1  1    )                     (     1 v2 v2 v2 )
 *         ( v1 v2  1 )                     (        1 v3 v3 )
 *         ( v1 v2 v3 )
 *         ( v1 v2 v3 )
 *
 *     direction = Backward and         direction = Backward and
 *     storeMode = Columnwise:             storeMode = Rowwise:
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
    class side_t, class trans_t, class direction_t, class storage_t >
int larfb(
    side_t side, trans_t trans,
    direction_t direction, storage_t storeMode,
    const matrixV_t& V, const matrixT_t& T,
    matrixC_t& C, matrixW_t& work )
{
    using idx_t = size_type< matrixV_t >;
    using pair  = pair<idx_t,idx_t>;

    // constants
    const type_t< matrixW_t > one( 1 );

    // constants
    const idx_t m = nrows(C);
    const idx_t n = ncols(C);
    const idx_t k = nrows(T);

    // check arguments
    tlapack_error_if(    side != Side::Left &&
                        side != Side::Right, -1 );
    tlapack_error_if(    trans != Op::NoTrans &&
                        trans != Op::ConjTrans &&
                        (
                            (trans != Op::Trans) ||
                            is_complex< type_t< matrixV_t > >::value
                        ), -2 );
    tlapack_error_if(    direction != Direction::Backward &&
                        direction != Direction::Forward, -3 );
    tlapack_error_if(    storeMode != StoreV::Columnwise &&
                        storeMode != StoreV::Rowwise, -4 );

    if( direction == Direction::Forward )
    {
        if( storeMode == StoreV::Columnwise )
            tlapack_error_if( access_denied( strictLower, read_policy(V) ), -5 );
        else
            tlapack_error_if( access_denied( strictUpper, read_policy(V) ), -5 );

        tlapack_error_if( access_denied( Uplo::Upper, read_policy(T) ), -6 );
    }
    else
    {
        tlapack_error_if( access_denied( dense, read_policy(V) ), -5 );

        tlapack_error_if( access_denied( Uplo::Lower, read_policy(T) ), -6 );
    }

    tlapack_error_if(    access_denied( dense, write_policy(C) ), -7 );
    tlapack_error_if(    access_denied( dense, write_policy(work) ), -8 );

    // Quick return
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    if( storeMode == StoreV::Columnwise ){
        if( direction == Direction::Forward ){
            if( side == Side::Left ){
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,k} );
                const auto V2 = rows( V, ( m > k ) ? pair{k,m} : pair{0,0} );
                auto C1 = rows( C, pair{0,k} );
                auto C2 = rows( C, ( m > k ) ? pair{k,m} : pair{0,0} );
                auto W  = slice( work, pair{0,k}, pair{0,n} );

                // W := C1
                lacpy( dense, C1, W );
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
                const auto V2 = rows( V, ( n > k ) ? pair{k,n} : pair{0,0} );
                auto C1 = cols( C, pair{0,k} );
                auto C2 = cols( C, ( n > k ) ? pair{k,n} : pair{0,0} );
                auto W  = slice( work, pair{0,m}, pair{0,k} );

                // W := C1
                lacpy( dense, C1, W );
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
            if( side == Side::Left ){
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                const auto V1 = rows( V, pair{0,m-k} );
                const auto V2 = rows( V, pair{m-k,m} );
                auto C1 = rows( C, pair{0,m-k} );
                auto C2 = rows( C, pair{m-k,m} );
                auto W  = slice( work, pair{0,k}, pair{0,n} );

                // W := C2
                lacpy( dense, C2, W );
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
                auto W  = slice( work, pair{0,m}, pair{0,k} );

                // W := C2
                lacpy( dense, C2, W );
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
        if( direction == Direction::Forward ){
            if( side == Side::Left ){
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,k} );
                const auto V2 = cols( V, ( m > k ) ? pair{k,m} : pair{0,0} );
                auto C1 = rows( C, pair{0,k} );
                auto C2 = rows( C, ( m > k ) ? pair{k,m} : pair{0,0} );
                auto W  = slice( work, pair{0,k}, pair{0,n} );

                // W := C1
                lacpy( dense, C1, W );
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
                const auto V2 = cols( V, ( n > k ) ? pair{k,n} : pair{0,0} );
                auto C1 = cols( C, pair{0,k} );
                auto C2 = cols( C, ( n > k ) ? pair{k,n} : pair{0,0} );
                auto W  = slice( work, pair{0,m}, pair{0,k} );

                // W := C1
                lacpy( dense, C1, W );
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
            if( side == Side::Left ){
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                const auto V1 = cols( V, pair{0,m-k} );
                const auto V2 = cols( V, pair{m-k,m} );
                auto C1 = rows( C, pair{0,m-k} );
                auto C2 = rows( C, pair{m-k,m} );
                auto W  = slice( work, pair{0,k}, pair{0,n} );

                // W := C2
                lacpy( dense, C2, W );
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
                auto W  = slice( work, pair{0,m}, pair{0,k} );

                // W := C2
                lacpy( dense, C2, W );
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
