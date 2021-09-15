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

/** Forms the triangular factor T of a real block reflector H of order n,
 * which is defined as a product of k elementary reflectors.
 *
 *               If direct = 'F', H = H_1 H_2 . . . H_k and T is upper triangular.
 *               If direct = 'B', H = H_k . . . H_2 H_1 and T is lower triangular.
 *
 *  If storeV = 'C', the vector which defines the elementary reflector
 *  H(i) is stored in the i-th column of the array V, and
 *
 *               H  =  I - V * T * V'
 *
 *  If storeV = 'R', the vector which defines the elementary reflector
 *  H(i) is stored in the i-th row of the array V, and
 *
 *               H  =  I - V' * T * V
 *
 *  The shape of the matrix V and the storage of the vectors which define
 *  the H(i) is best illustrated by the following example with n = 5 and
 *  k = 3. The elements equal to 1 are not stored.
 *
 *               direct='F' & storeV='C'          direct='F' & storeV='R'
 *               -----------------------          -----------------------
 *               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
 *                   ( v1  1    )                     (     1 v2 v2 v2 )
 *                   ( v1 v2  1 )                     (        1 v3 v3 )
 *                   ( v1 v2 v3 )
 *                   ( v1 v2 v3 )
 *
 *               direct='B' & storeV='C'          direct='B' & storeV='R'
 *               -----------------------          -----------------------
 *               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
 *                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
 *                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
 *                   (     1 v3 )
 *                   (        1 )
 *
 * @return 0 if success.
 * @return -i if the ith argument is invalid.
 * @tparam real_t Floating point type.
 * @param direct Specifies the direction in which the elementary reflectors are multiplied to form the block reflector.
 *
 *               'F' : forward
 *               'B' : backward
 *
 * @param storeV Specifies how the vectors which define the elementary reflectors are stored.
 *
 *               'C' : columnwise
 *               'R' : rowwise
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
template <typename TV>
int larft(
    Direction direct, StoreV storeV,
    idx_t n, idx_t k,
    TV *V, idx_t ldV,
    TV *tau, TV *T, idx_t ldT)
{
    typedef blas::real_type<TV> real_t;
    using std::conj;
    using std::max;
    using std::min;
    using std::toupper;

    // constants
    const TV one(1.0);
    const TV zero(0.0);

    // check arguments
    lapack_error_if( direct != Direction::Forward &&
                     direct != Direction::Backward, -1 );
    lapack_error_if( storeV != StoreV::Columnwise &&
                     storeV != StoreV::Rowwise, -2 );
    lapack_error_if( n < 0, -3 );
    lapack_error_if( k < 1, -4 );
    lapack_error_if( ldV < ((storeV == StoreV::Columnwise) ? n : k), -6 );
    lapack_error_if( ldT < k, -9 );

    if (n == 0)
        return 0;

    TV *V0 = V;
    TV *T0 = T;
    if (direct == 'F')
    {
        for (idx_t i = 0; i < k; i++)
        {
            if (tau[i] == zero)
            {
                for (int j = 0; j <= i; j++)
                    T[j] = zero;
            }
            else
            {
                if (storeV == 'C')
                {
                    TV *v = V0;
                    for (idx_t j = 0; j < i; j++)
                    {
                        T[j] = -tau[i] * conj(v[i]);
                        v += ldV;
                    }
                    GEMV<real_t>('C', n - i - 1, i, -tau[i], &V0[i + 1], ldV, &V[i + 1], 1, one, T, 1);
                }
                else // storeV=='R'
                {
                    for (idx_t j = 0; j < i; j++)
                        T[j] = -tau[i] * V[j];
                    GEMM<real_t>('N', 'C', i, 1, n - i - 1, -tau[i], V + ldV, ldV, V + i + ldV, ldV, one, T, ldT);
                }
                TRMV<real_t>('U', 'N', 'N', i, T0, ldT, T, 1);
                T[i] = tau[i];
            }
            T += ldT;
            V += ldV;
        }
    }
    else // direct=='B'
    {
        T += (k - 1) * ldT;
        V += (k - 1) * ldV;
        T[k - 1] = tau[k - 1];
        for (idx_t i = k - 2; i >= 0; --i)
        {
            T -= ldT;
            V -= ldV;
            if (tau[i] == zero)
            {
                for (idx_t j = i; j < k; j++)
                    T[j] = zero;
            }
            else
            {
                if (storeV == 'C')
                {
                    TV *v = V + ldV;
                    for (idx_t j = i + 1; j < k; j++)
                    {
                        T[j] = -tau[i] * conj(v[n - k + i]);
                        v += ldV;
                    }
                    GEMV<real_t>('C', n - k + i, k - i - 1, -tau[i], V + ldV, ldV, V, 1, one, T + i + 1, 1);
                }
                else // storeV=='R'
                {
                    TV *v = V + (n - k) * ldV;
                    for (idx_t j = i + 1; j < k; j++)
                    {
                        T[j] = -tau[i] * v[j];
                    }
                    GEMM<real_t>('N', 'C', k - i - 1, 1, n - k + i, -tau[i], V0 + i + 1, ldV, V0 + i, ldV, one, T + i + 1, ldT);
                }
                TRMV<real_t>('L', 'N', 'N', k - i - 1, T + i + 1 + ldT, ldT, T + i + 1, 1);
            }
            T[i] = tau[i];
        }
    }
    return 0;
}

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @param work Vector of size n (m if side == Side::Right).
 * @see larft( Side, Op, blas::idx_t, blas::idx_t, blas::idx_t, const TA*, blas::idx_t, const blas::real_type<TA,TC>*, TC*, blas::idx_t )
 * 
 * @ingroup geqrf
 */
template<typename TA, typename TC>
int larft(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    const TA* A, blas::idx_t lda,
    const blas::real_type<TA,TC>* tau,
    TC* C, blas::idx_t ldc,
    blas::scalar_type<TA,TC>* work )
{
    // check arguments

    lapack_error_if( side != Side::Left &&
                     side != Side::Right, -1 );
    lapack_error_if( trans != Op::NoTrans &&
                     trans != Op::Trans &&
                     trans != Op::ConjTrans, -2 );
    lapack_error_if( m < 0, -3 );
    lapack_error_if( n < 0, -4 );

    const idx_t q = (side == Side::Left) ? m : n;
    lapack_error_if( k < 0 || k > q, -5 );
    lapack_error_if( lda < q, -7 );
    lapack_error_if( ldc < m, -10 );

    // quick return
    if ((m == 0) || (n == 0) || (k == 0))
        return 0;

    if( side == Side::Left ) {
        if( trans == Op::NoTrans ) {
            for (idx_t i = 0; i < k; ++i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (idx_t i = k-1; i != idx_t(-1); --i)
                larf( Side::Left, m-k+i+1, n, A+i, lda, tau[i], C, ldc, work );
        }
    }
    else { // side == Side::Right
        if( trans == Op::NoTrans ) {
            for (idx_t i = 0; i < k; ++i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
        else {
            for (idx_t i = k-1; i != idx_t(-1); --i)
                larf( Side::Right, m, n-k+i+1, A+i, lda, tau[i], C, ldc, work );
        }
    }

    return 0;
}

/** Applies orthogonal matrix Q to a matrix C.
 * 
 * @return  0 if success
 * @return -i if the ith argument is invalid
 * 
 * @param[in] side Specifies which side Q is to be applied.
 *                 'L': apply Q or Q' from the Left;
 *                 'R': apply Q or Q' from the Right.
 * @param[in] trans Specifies whether Q or Q' is applied.
 *                 'N':  No transpose, apply Q;
 *                 'T':  Transpose, apply Q'.
 * @param[in] m The number of rows of the matrix C.
 * @param[in] n The number of columns of the matrix C.
 * @param[in] k The number of elementary reflectors whose product defines the matrix Q.
 *                 If side='L', m>=k>=0;
 *                 if side='R', n>=k>=0.
 * @param[in] A Matrix containing the elementary reflectors H.
 *                 If side='L', A is k-by-m;
 *                 if side='R', A is k-by-n.
 * @param[in] ldA The column length of the matrix A.  ldA>=k.
 * @param[in] tau Real vector of length k containing the scalar factors of the
 * elementary reflectors.
 * @param[in,out] C m-by-n matrix. 
 *     On exit, C is replaced by one of the following:
 *                 If side='L' & trans='N':  C <- Q * C
 *                 If side='L' & trans='T':  C <- Q'* C
 *                 If side='R' & trans='T':  C <- C * Q'
 *                 If side='R' & trans='N':  C <- C * Q
 * @param ldC The column length the matrix C. ldC>=m.
 * 
 * @ingroup geqrf
 */
template<typename TA, typename TC>
inline int larft(
    Side side, Op trans,
    blas::idx_t m, blas::idx_t n, blas::idx_t k,
    const TA* A, blas::idx_t lda,
    const blas::real_type<TA,TC>* tau,
    TC* C, blas::idx_t ldc )
{
    typedef blas::scalar_type<TA,TC> scalar_t;

    int info = 0;
    scalar_t* work = new scalar_t[
        (side == Side::Left)
            ? ( (m >= 0) ? m : 0 )
            : ( (n >= 0) ? n : 0 )
    ];

    info = larft( side, trans, m, n, k, A, lda, tau, C, ldc, work );

    delete[] work;
    return info;
}

}

#endif // __LARFT_HH__