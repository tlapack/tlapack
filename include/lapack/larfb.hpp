/// @file larfb.hpp Applies a Householder block reflector to a matrix.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// Adapted from @see https://github.com/langou/latl/blob/master/include/larfb.h
//
// Copyright (c) 2013-2021, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __LARFB_HH__
#define __LARFB_HH__

#include "lapack/types.hpp"
#include "lapack/lacpy.hpp"

#include "tblas.hpp"
#include "slate_api/blas.hpp"

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
template <typename TV, typename TC>
int larfb(
    Side side, Op trans,
    Direction direct, StoreV storeV,
    idx_t m, idx_t n, idx_t k,
    TV const* V, idx_t ldV,
    TV const* T, idx_t ldT,
    TC* C, idx_t ldC,
    blas::scalar_type<TV, TC> *W )
{
    typedef blas::real_type<TV, TC> real_t;
    using blas::conj;

    // constants
    const real_t one(1.0);

    // check arguments
    if( blas::is_complex<TV>::value )
        lapack_error_if( trans == Op::Trans, -2 );

    // Quick return
    if (m <= 0 || n <= 0) return 0;

    if (storeV == StoreV::Columnwise) {
        if (direct == Direction::Forward) {
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+k;
                TC* C1 = C;
                TC* C2 = C+k;

                // W := C1
                lacpy(Uplo::General,k,n,C1,ldC,W,k);
                // W := V1^H W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    k, n, one, V1, ldV, W, k);
                if( m > k )
                    // W := W + V2^H C2
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        k, n, m-k, one, V2, ldV,
                        C2, ldC, one, W, k);
                // W := op(T) W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    // C2 := C2 - V2 W
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m-k, n, k, -one, V2, ldV,
                        W, k, one, C2, ldC);
                // W := - V1 W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, -one, V1, ldV, W, k);

                // C1 := C1 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C1[ i + j*ldC ] += W[ i + j*k ];
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+k;
                TC* C1 = C;
                TC* C2 = C+k*ldC;

                // W := C1
                lacpy(Uplo::General,m,k,C1,ldC,W,m);
                // W := W V1
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    m, k, one, V1, ldV, W, m);
                if( n > k )
                    // W := W + C2 V2
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, k, n-k, one,
                        C2, ldC, V2, ldV, one, W, m);
                // W := W op(T)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    // C2 := C2 - W V2^H
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m, n-k, k, -one, W, m,
                        V2, ldV, one, C2, ldC);
                // W := - W V1^H
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    m, k, -one, V1, ldV, W, m);
                
                // C1 := C1 + W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C1[ i + j*ldC ] += W[ i + j*m ];
            }
        }
        else { // direct == Direction::Backward
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an m-by-k matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+m-k;
                TC* C1 = C;
                TC* C2 = C+m-k;

                // W := C2
                lacpy(Uplo::General,k,n,C2,ldC,W,k);
                // W := V2^H W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    k, n, one, V2, ldV, W, k);
                if( m > k )
                    // W := W + V1^H C1
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        k, n, m-k, one, V1, ldV,
                        C1, ldC, one, W, k);
                // W := op(T) W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    // C1 := C1 - V1 W
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m-k, n, k, -one, V1, ldV,
                        W, k, one, C1, ldC);
                // W := - V2 W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    k, n, -one, V2, ldV, W, k);

                // C2 := C2 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C2[ i + j*ldC ] += W[ i + j*k ];
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an n-by-k matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+n-k;
                TC* C1 = C;
                TC* C2 = C+(n-k)*ldC;

                // W := C(0:m,n-k:n)
                lacpy(Uplo::General,m,k,C2,ldC,W,m);
                // W := W V(n-k:n,0:k)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    m, k, one, V2, ldV, W, m);
                if( n > k )
                    // W := W + C(0:m,0:n-k) V(0:n-k,0:k)
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, k, n-k, one,
                        C1, ldC, V1, ldV, one, W, m);
                // W := W op(T)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    // C(0:m,0:n-k) := C(0:m,0:n-k) - W V(0:n-k,0:k)^H
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m, n-k, k, -one, W, m,
                        V1, ldV, one, C1, ldC);
                // W := - W V(n-k:n,0:k)^H
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    m, k, -one, V2, ldV, W, m);
                
                // C(0:m,n-k:n) := W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C2[ i + j*ldC ] += W[ i + j*m ];
            }
        }
    }
    else { // storeV == StoreV::Rowwise
        if (direct == Direction::Forward) {
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+k*ldV;
                TC* C1 = C;
                TC* C2 = C+k;

                // W := C1
                lacpy(Uplo::General,k,n,C1,ldC,W,k);
                // W := V1 W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    k, n, one, V1, ldV, W, k);
                if( m > k )
                    // W := W + V2 C2
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        k, n, m-k, one, V2, ldV,
                        C2, ldC, one, W, k);
                // W := op(T) W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    // C2 := C2 - V2^H W
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        m-k, n, k, -one, V2, ldV,
                        W, k, one, C2, ldC);
                // W := - V1^H W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    k, n, -one, V1, ldV, W, k);

                // C1 := C1 - W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C1[ i + j*ldC ] += W[ i + j*k ];
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+k*ldV;
                TC* C1 = C;
                TC* C2 = C+k*ldC;

                // W := C(0:m,0:k)
                lacpy(Uplo::General,m,k,C1,ldC,W,m);
                // W := W V(0:k,0:k)^H
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    m, k, one, V1, ldV, W, m);
                if( n > k )
                    // W := W + C(0:m,k:n) V(0:k,k:n)^H
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m, k, n-k, one,
                        C2, ldC, V2, ldV, one, W, m);
                // W := W op(T)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    // C(0:m,k:n) := C(0:m,k:n) - W V(0:k,k:n)
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, n-k, k, -one, W, m,
                        V2, ldV, one, C2, ldC);
                // W := - W V(0:k,0:k)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    m, k, -one, V1, ldV, W, m);
                
                // C(0:m,0:k) := W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C1[ i + j*ldC ] += W[ i + j*m ];
            }
        }
        else { // direct == Direction::Backward
            if (side == Side::Left) {
                // W is an k-by-n matrix
                // V is an k-by-m matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+(m-k)*ldV;
                TC* C1 = C;
                TC* C2 = C+m-k;

                // W := C2
                lacpy(Uplo::General,k,n,C2,ldC,W,k);
                // W := V2 W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, one, V2, ldV, W, k);
                if( m > k )
                    // W := W + V1 C1
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        k, n, m-k, one, V1, ldV,
                        C1, ldC, one, W, k);
                // W := op(T) W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    // C1 := C1 - V1^H W
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        m-k, n, k, -one, V1, ldV,
                        W, k, one, C1, ldC);
                // W := - V2^H W
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    k, n, -one, V2, ldV, W, k);

                // C2 := C2 + W
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        C2[ i + j*ldC ] += W[ i + j*k ];
            }
            else { // side == Side::Right
                // W is an m-by-k matrix
                // V is an k-by-n matrix

                // Matrix views
                TV const* V1 = V;
                TV const* V2 = V+(n-k)*ldV;
                TC* C1 = C;
                TC* C2 = C+(n-k)*ldC;

                // W := C(0:m,n-k:n)
                lacpy(Uplo::General,m,k,C2,ldC,W,m);
                // W := W V(0:k,n-k:n)^H
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    m, k, one, V2, ldV, W, m);
                if( n > k )
                    // W := W + C(0:m,0:n-k) V(0:k,0:n-k)^H
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m, k, n-k, one,
                        C1, ldC, V1, ldV, one, W, m);
                // W := W op(T)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    // C(0:m,0:n-k) := C(0:m,0:n-k) - W V(0:k,0:n-k)
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, n-k, k, -one, W, m,
                        V1, ldV, one, C1, ldC);
                // W := - W V(0:k,n-k:n)
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    m, k, -one, V2, ldV, W, m);
                
                // C(0:m,n-k:n) := W
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        C2[ i + j*ldC ] += W[ i + j*m ];
            }
        }
    }

    return 0;
}

}

#endif // __LARFB_HH__
