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

namespace lapack {

/** Applies a block reflector $H$ or its transpose $H^H$ to a
 * m-by-n matrix C, from either the left or the right.
 *
 * @param[in] side
 *     - lapack::Side::Left:  apply $H$ or $H^H$ from the Left
 *     - lapack::Side::Right: apply $H$ or $H^H$ from the Right
 *
 * @param[in] trans
 *     - lapack::Op::NoTrans:   apply $H  $ (No transpose)
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
void larfb(
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

    // Quick return
    if (m <= 0 || n <= 0) return;

    #define _C(i_, j_) C[ (i_) + (j_)*ldC ]

    if (storeV == StoreV::Columnwise) {
        if (direct == Direction::Forward) {
            if (side == Side::Left) {
                lacpy(Uplo::General,k,n,C,ldC,W,k);
                
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    k, n, one, V, ldV, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        k, n, m-k, one, V+k, ldV,
                        C+k, ldC, one, W, k);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m-k, n, k, -one, V+k, ldV,
                        W, k, one, C+k, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, -one, V, ldV, W, k);
                
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        _C(i,j) -= conj( W[ i + j*k ] );
            }
            else { // side == Side::Right
                lacpy(Uplo::General,m,k,C,ldC,W,m);
                
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    m, k, one, V, ldV, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m , k, n-k, one,
                        C+k*ldC, ldC, V+k, ldV, one, W, m);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m-k, n, k, -one, V+k, ldV,
                        W, m, one, C+k, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    n, k, -one, V, ldV, W, m);
                
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < n; ++i )
                        _C(i,j) -= conj( W[ i + j*m ] );
            }
        }
        else { // direct == Direction::Backward
            if (side == Side::Left) {
                lacpy(Uplo::General,k,n,C+m-k,ldC,W,k);
                
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    k, n, one, V+m-k, ldV, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        k, n, m-k, one, V, ldV,
                        C, ldC, one, W, k);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m-k, n, k, -one, V, ldV,
                        W, k, one, C, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, -one, V+m-k, ldV, W, k);
                
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        _C(i+m-k,j) -= conj( W[ i + j*k ] );
            }
            else { // side == Side::Right
                lacpy(Uplo::General,m,k,C+(n-k)*ldC,ldC,W,m);
                
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    m, k, one, V+n-k, ldV, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m , k, n-k, one,
                        C, ldC, V, ldV, one, W, m);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m-k, n, k, -one, V, ldV,
                        W, m, one, C+k, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    n, k, -one, V+n-k, ldV, W, m);
                
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < n; ++i )
                        _C(i,j+(n-k)) -= conj( W[ i + j*m ] );
            }
        }
    }
    else { // storeV == StoreV::Rowwise
        if (direct == Direction::Forward) {
            if (side == Side::Left) {
                lacpy(Uplo::General,k,n,C,ldC,W,k);
                
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    k, n, one, V, ldV, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        k, n, m-k, one, V+k*ldV, ldV,
                        C+k, ldC, one, W, k);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        m-k, n, k, -one, V+k*ldV, ldV,
                        W, k, one, C+k, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    k, n, -one, V, ldV, W, k);
                
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        _C(i,j) -= conj( W[ i + j*k ] );
            }
            else { // side == Side::Right
                lacpy(Uplo::General,m,k,C,ldC,W,m);
                
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::ConjTrans, Diag::Unit,
                    m, k, one, V, ldV, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m , k, n-k, one,
                        C+k*ldC, ldC, V+k*ldV, ldV, one, W, m);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, n-k, k, -one, W, m,
                        V+k*ldV, ldV, one, C+k*ldC, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Upper, Op::NoTrans, Diag::Unit,
                    m, k, -one, V, ldV, W, m);
                
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < m; ++i )
                        _C(i,j) -= conj( W[ i + j*m ] );
            }
        }
        else { // direct == Direction::Backward
            if (side == Side::Left) {
                lacpy(Uplo::General,k,n,C+m-k,ldC,W,k);
                
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, one, V+(m-k)*ldV, ldV, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        k, n, m-k, one, V, ldV,
                        C, ldC, one, W, k);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, trans, Diag::NonUnit,
                    k, n, one, T, ldT, W, k);
                if( m > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m-k, n, k, -one, V, ldV,
                        W, k, one, C, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Left,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    k, n, -one, V+(m-k)*ldV, ldV, W, k);
                
                for( idx_t j = 0; j < n; ++j )
                    for( idx_t i = 0; i < k; ++i )
                        _C(i+m-k,j) -= conj( W[ i + j*k ] );
            }
            else { // side == Side::Right
                lacpy(Uplo::General,m,k,C+(n-k)*ldC,ldC,W,m);
                
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::ConjTrans, Diag::Unit,
                    m, k, one, V+(n-k)*ldV, ldV, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::ConjTrans,
                        m , k, n-k, one,
                        C, ldC, V, ldV, one, W, m);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, trans, Diag::NonUnit,
                    m, k, one, T, ldT, W, m);
                if( n > k )
                    blas::gemm(
                        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, n-k, k, -one, W, m,
                        V, ldV, one, C, ldC);
                blas::trmm(
                    Layout::ColMajor, Side::Right,
                    Uplo::Lower, Op::NoTrans, Diag::Unit,
                    m, k, -one, V+(n-k)*ldV, ldV, W, m);
                
                for( idx_t j = 0; j < k; ++j )
                    for( idx_t i = 0; i < n; ++i )
                        _C(i,j+(n-k)) -= conj( W[ i + j*m ] );
            }
        }
    }

    #undef _C
}


template <typename TV, typename TC>
inline void larfb(
    Side side, Op trans,
    Direction direct, StoreV storeV,
    idx_t m, idx_t n, idx_t k,
    TV const* V, idx_t ldV,
    TV const* T, idx_t ldT,
    TC* C, idx_t ldC )
{
    typedef blas::scalar_type<TV, TC> scalar_t;
    scalar_t *work = new scalar_t[ k*m ];

    larfb(  side, trans, direct, storeV,
            m, n, k, V, ldV, T, ldT, C, ldC, work );

    delete[] work;
}

}

#endif // __LARFB_HH__